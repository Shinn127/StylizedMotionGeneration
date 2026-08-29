import re
import numpy as np


channelmap = {
    "Xrotation": "x",
    "Yrotation": "y",
    "Zrotation": "z",
}

channelmap_inv = {
    "x": "Xrotation",
    "y": "Yrotation",
    "z": "Zrotation",
}

ordermap = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def load(filename, order=None):
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    joint_channels: list[int] = []

    for line in f:
        if "HIERARCHY" in line:
            continue
        if "MOTION" in line:
            continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            joint_channels.append(-1)
            active = len(parents) - 1
            continue

        if "{" in line:
            continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if joint_channels and 0 <= active < len(joint_channels):
                joint_channels[active] = channels
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis : 2 + channelie]
                if any(p not in channelmap for p in parts):
                    continue
                order = "".join(channelmap[p] for p in parts)
            continue

        jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            joint_channels.append(-1)
            active = len(parents) - 1
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        dmatch = line.strip().split(" ")
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            n_bones = len(parents)
            fi = i
            # ``channels`` holds the last-seen CHANNELS declaration; files with
            # a uniform layout match it exactly, while mixed layouts (e.g.
            # soma_uniform motions: 6-channel Root and Hips plus 3-channel
            # joints) fall through to the per-joint walk below.
            uniform_three = channels == 3 and data_block.shape[0] == 3 + 3 * n_bones
            uniform_six = channels == 6 and data_block.shape[0] == 6 * n_bones
            uniform_nine = channels == 9 and data_block.shape[0] == 3 + 9 * (n_bones - 1)
            mixed_total = sum(3 if count == 3 else 6 for count in joint_channels)
            if uniform_three:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(n_bones, 3)
            elif uniform_six:
                data_block = data_block.reshape(n_bones, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif uniform_nine:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(n_bones - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            elif data_block.shape[0] == mixed_total and -1 not in joint_channels:
                # Each joint consumes its own channel count in file order.
                cursor = 0
                for bone, joint_channel_count in enumerate(joint_channels):
                    if joint_channel_count == 3:
                        rotations[fi, bone] = data_block[cursor : cursor + 3]
                        cursor += 3
                    else:
                        positions[fi, bone] = data_block[cursor : cursor + 3]
                        rotations[fi, bone] = data_block[cursor + 3 : cursor + 6]
                        cursor += 6
            else:
                raise Exception(f"Unsupported channel layout: {data_block.shape[0]} values for {n_bones} bones")

            i += 1

    f.close()

    return {
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
    }


def read_frame_count(filename):
    with open(filename, "r") as f:
        for line in f:
            fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
            if fmatch:
                return int(fmatch.group(1))
    raise ValueError(f"Missing Frames header in BVH file: {filename}")
