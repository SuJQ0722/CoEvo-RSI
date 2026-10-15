"""Fixed official HomeBench four-shot in-context demonstrations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Keep this text aligned with data/HomeBench/code/four_shot.txt.  In
# particular, preserve the released prompt's spelling and argument syntax.
OFFICIAL_FOUR_SHOT = """Here are a few examples, your output format should be consistent with the results provided in the example:
<example1>
    <User instructions:> Lower the air conditioner temperature in the guest bedroom by 1 degree.
    <Machine instructions:> {guest_bedroom.air_conditione.set_temperature(26)}
</example1>
<example2>
    <User instructions:> Turn on the light in the living room.
    <Machine instructions:> {error_input}  (If there is no device or if there is no attribute or method that can be operated on the device, please output as "error_input".)
</example2>
<example3>
    <User instructions:> Set the intensity of the humidifier to 20% in the guest bedroom, adjust the intensity of the dehumidifiers by increasing it by 30% in the study room, and set the intensity of the dehumidifiers to 40% in the study room.
    <Machine instructions:> {guest_bedroom.humidifier.set_intensity(20),study_room.dehumidifiers.set_intensity(10),study_room.dehumidifiers.set_intensity(40),}
</example3>
<example4>
    <User instructions:> Increase the intensity of the aromatherapy in the corridor by 4%, set the brightness of the lights in the garage to 40, adjust the fan speed of the air conditioner in the living room to low, and set the volume of the media player in the living room to 0.
    <Machine instructions:> {corridor.aromatherapy.set_intensity(30),error_input,error_input,living_room.media_player.set_volume(0),}
</example4>"""


def inject_official_four_shot(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inject the fixed demonstrations immediately before the evaluated request."""

    result = deepcopy(messages)
    if any(OFFICIAL_FOUR_SHOT in str(message.get("content") or "") for message in result):
        return result

    request_marker = "\n<User instructions>\n"
    for message in reversed(result):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and request_marker in content:
            prefix, request = content.rsplit(request_marker, 1)
            message["content"] = (
                prefix + "\n\n" + OFFICIAL_FOUR_SHOT + request_marker + request
            )
            return result

    for message in result:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            message["content"] = message["content"].rstrip() + "\n\n" + OFFICIAL_FOUR_SHOT
            return result

    return [{"role": "system", "content": OFFICIAL_FOUR_SHOT}, *result]
