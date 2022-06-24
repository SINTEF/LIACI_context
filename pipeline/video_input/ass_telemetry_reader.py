import json
import ass
import datetime
import pandas as pd
import numpy as np

def get_telemetry_data(ass_file_name):
    jsondata = []
    with open(ass_file_name, encoding='utf_8_sig') as f:
        doc = ass.parse(f)
        entry_start = doc.events[0].start        
        
        # get the number of relevant lines
        delta_step = 0
        for entry in doc.events:                    
            if entry_start != entry.start:  
                break; 
            entry_start = entry.start
            delta_step += 1  

        
        for i in range(0,len(doc.events),delta_step): 
            n = {   'time': [],
                    'data': {}}
            
            keys = doc.events[i].text.split('}')[1].replace(':','').split('\\N')
            values_with_unit = doc.events[i+1].text.split('}')[1].split('\\N')
            for key, val in zip(keys,values_with_unit):
                unit = ''             
                if val.find(' ') >= 0:
                    unit = val.split(' ')[1]  
                value =  val.split(' ')[0]
                n['data'][key] = {'value': value,'unit': unit}

            keys = doc.events[i+2].text.split('}')[1].replace(':','').split('\\N')
            values_with_unit = doc.events[i+3].text.split('}')[1].split('\\N')
            for key, val in zip(keys,values_with_unit):   
                unit = ''             
                if val.find(' ') >= 0:
                    unit = val.split(' ')[1]                   
                value =  val.split(' ')[0]
                n['data'][key] = {'value': value,'unit': unit}

            keys = doc.events[i+4].text.split('}')[1].replace(':','').split('\\N')
            values_with_unit = doc.events[i+5].text.split('}')[1].split('\\N')
            for key, val in zip(keys,values_with_unit):   
                unit = ''             
                if val.find(' ') >= 0:
                    unit = val.split(' ')[1]                   
                value =  val.split(' ')[0]
                n['data'][key] = {'value': value,'unit': unit}

            date_format=datetime.datetime.strptime(n['data']['Date']['value']+'-'+n['data']['Time']['value'], '%d.%m.%Y-%H:%M:%S')
            n['time'] = date_format
            jsondata.append(n)

    return jsondata
    
def interpolate_for_all_frames(telemetry_json, num_of_frames):
    keys_to_interpolate = ['Pitch', 'Roll', 'Camera Tilt', 'Heading',
    'Depth', 'Rangefinder', 'Longitude','Latitude']
    telemetry_dict = {'time': [],}
    
    for entry in telemetry_json:        
        telemetry_dict['time'].append(entry['time'])        
        for key, value in entry['data'].items():
            if key not in keys_to_interpolate:
                continue           
            if key in telemetry_dict.keys():
                telemetry_dict[key].append(float(value['value']))
            else:
                telemetry_dict[key] = [float(value['value'])]

    df = pd.DataFrame.from_dict(telemetry_dict)
    df = df.set_index('time')
    upsampled_array = pd.date_range(start=telemetry_dict['time'][0], end=telemetry_dict['time'][-1], periods=num_of_frames)
    t = df.index
    df = df[~df.index.duplicated()]
    interpolated = df.reindex(t.union(upsampled_array)).interpolate('index').loc[upsampled_array]
    interpolated.insert(0, 'frame_index', np.arange(0, num_of_frames, dtype=int))

    return interpolated

def read_telemetry_data(ass_file, frame_count):
    telemetry_json = get_telemetry_data(ass_file)
    telemetry_df = interpolate_for_all_frames(telemetry_json, frame_count)
    return telemetry_df #json.loads(telemetry_df.to_json())