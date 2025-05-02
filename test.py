from unidepth.models import UniDepthV2

name = "unidepth-v2-vits14"

model_v2 = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
