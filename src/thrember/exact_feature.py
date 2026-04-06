import json
from thrember.features import PEFeatureExtractor

pe_path = r"D:\laptrinh\python\project2\EMBER2024\src\thrember\jdk-21_windows-x64_bin.exe"

with open(pe_path, "rb") as f:
    bytez = f.read()

extractor = PEFeatureExtractor()
raw_feat = extractor.raw_features(bytez)

with open("sample_raw_features.json", "w", encoding="utf-8") as f:
    json.dump(raw_feat, f, indent=2, ensure_ascii=False)

print("Đã lưu raw features vào sample_raw_features.json")