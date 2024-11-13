import xml.etree.ElementTree as ET


def parse_tripinfo_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    tripinfo_dict = {}

    for tripinfo in root.findall('tripinfo'):
        trip_id = tripinfo.get('id')

        # Extract tripinfo attributes
        tripinfo_data = {attr: tripinfo.get(attr) for attr in tripinfo.attrib}

        # Extract emissions attributes if present and merge with tripinfo_data
        emissions = tripinfo.find('emissions')
        if emissions is not None:
            emissions_data = {attr: emissions.get(attr) for attr in emissions.attrib}
            tripinfo_data.update(emissions_data)  # Merge emissions data into tripinfo_data

        # Store the combined data in the dictionary with trip_id as the key
        tripinfo_dict[trip_id] = tripinfo_data

    return tripinfo_dict


# 示例文件路径
file_path = 'tripinfo.xml'

# 解析XML文件并获取字典
tripinfo_dict = parse_tripinfo_file(file_path)
first_key_value_pair = next(iter(tripinfo_dict.items()))
print("First key-value pair:", first_key_value_pair)

total_co2 = 0.0
total_time = 0.0
total_waitingTime = 0.0
total_time_loss = 0.0
total_fuel = 0.0
# 打印结果
for trip_id, data in tripinfo_dict.items():
    # print(f"Trip ID: {trip_id}")
    # print("Data:", data)
    total_fuel += float(data['fuel_abs'])
    total_time += float(data['duration'])
    total_waitingTime += float(data['waitingTime'])
    total_co2 += float(data['CO2_abs'])
    total_time_loss += float(data['timeLoss'])
print(total_time_loss / len(tripinfo_dict), total_fuel / len(tripinfo_dict) / 1000)
print(total_time / len(tripinfo_dict), total_waitingTime / len(tripinfo_dict), total_co2 / len(tripinfo_dict) / 1000)
