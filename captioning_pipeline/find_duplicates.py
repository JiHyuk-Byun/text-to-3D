from collections import defaultdict
import pickle as pkl

pkl_path = input("Type pkl path: ")

# 중복 체크 함수
def find_duplicates(input_dict):
    # 문자열을 key로 사용하고, 해당 문자열이 등장하는 키 리스트를 저장
    value_map = defaultdict(list)
    
    # 각 value를 key로, 위치를 list로 저장
    for k, v in input_dict.items():
        value_map[v[0]].append(k)
    
    # 중복된 문자열 찾기
    duplicates = {k: v for k, v in value_map.items() if len(v) > 1}
    
    if duplicates:
        print("중복된 문자열이 있습니다:")
        for value, keys in duplicates.items():
            print(f"'{value}'이(가) {keys} 키들에 존재합니다.")
    else:
        print("중복된 문자열이 없습니다.")


with open(pkl_path, "rb") as fr:
    data = pkl.load(fr)

find_duplicates(data)
