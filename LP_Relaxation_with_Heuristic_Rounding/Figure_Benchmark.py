import pandas as pd
import matplotlib.pyplot as plt
import io
import re


cutting_plane_results =  {'LA01': 1106.0, 'LA02': 1308.0, 'LA03': 917.0, 'LA04': 1181.0, 'LA05': 1026.0, 'LA06': 1634.0, 'LA07': 1348.0, 'LA08': 1397.0, 'LA09': 1581.0, 'LA10': 1683.0, 'LA11': 1886.0, 'LA12': 1781.0, 'LA13': 1849.0, 'LA14': 2201.0, 'LA15': 2233.0, 'LA16': 1477.0, 'LA17': 1329.0, 'LA18': 1665.0, 'LA19': 1713.0, 'LA20': 1473.0, 'LA21': 1708.0, 'LA22': 2058.0, 'LA23': 2652.0, 'LA24': 2077.0, 'LA25': 1904.0, 'LA26': 2927.0, 'LA27': 2799.0, 'LA28': 2625.0, 'LA29': 2776.0, 'LA30': 2514.0, 'LA31': 3576.0, 'LA32': 3911.0, 'LA33': 3941.0, 'LA34': 3622.0, 'LA35': 4198.0, 'LA36': 2928.0, 'LA37': 2844.0, 'LA38': 1925.0, 'LA39': 2467.0, 'LA40': 2773.0}

SA_Results = {'LA01': 666, 'LA02': 667, 'LA03': 661, 'LA04': 597, 'LA05': 593, 'LA06': 926, 'LA07': 890, 'LA08': 863, 'LA09': 951, 'LA10': 958, 'LA11': 1222, 'LA12': 1039, 'LA13': 1150, 'LA14': 1292, 'LA15': 1289, 'LA16': 991, 'LA17': 821, 'LA18': 872, 'LA19': 872, 'LA20': 959, 'LA21': 1159, 'LA22': 1002, 'LA23': 1072, 'LA24': 1012, 'LA25': 1079, 'LA26': 1274, 'LA27': 1369, 'LA28': 1383, 'LA29': 1337, 'LA30': 1453, 'LA31': 1816, 'LA32': 1860, 'LA33': 1741, 'LA34': 1829, 'LA35': 1940, 'LA36': 1341, 'LA37': 1525, 'LA38': 1289, 'LA39': 1392, 'LA40': 1306}

CP_SAT_data_list = [ {'dataset': 'la01', 'makespan': 666}, {'dataset': 'la02', 'makespan': 655}, {'dataset': 'la03', 'makespan': 597}, {'dataset': 'la04', 'makespan': 590}, {'dataset': 'la05', 'makespan': 593}, {'dataset': 'la06', 'makespan': 926}, {'dataset': 'la07', 'makespan': 890}, {'dataset': 'la08', 'makespan': 863}, {'dataset': 'la09', 'makespan': 951}, {'dataset': 'la10', 'makespan': 958}, {'dataset': 'la11', 'makespan': 1222}, {'dataset': 'la12', 'makespan': 1039}, {'dataset': 'la13', 'makespan': 1150}, {'dataset': 'la14', 'makespan': 1292}, {'dataset': 'la15', 'makespan': 1207}, {'dataset': 'la16', 'makespan': 945}, {'dataset': 'la17', 'makespan': 784}, {'dataset': 'la18', 'makespan': 848}, {'dataset': 'la19', 'makespan': 842}, {'dataset': 'la20', 'makespan': 902}, {'dataset': 'la21', 'makespan': 1046}, {'dataset': 'la22', 'makespan': 927}, {'dataset': 'la23', 'makespan': 1032}, {'dataset': 'la24', 'makespan': 935}, {'dataset': 'la25', 'makespan': 977}, {'dataset': 'la26', 'makespan': 1218}, {'dataset': 'la27', 'makespan': 1235}, {'dataset': 'la28', 'makespan': 1216}, {'dataset': 'la29', 'makespan': 1152}, {'dataset': 'la30', 'makespan': 1355}, {'dataset': 'la31', 'makespan': 1784}, {'dataset': 'la32', 'makespan': 1850}, {'dataset': 'la33', 'makespan': 1719}, {'dataset': 'la34', 'makespan': 1721}, {'dataset': 'la35', 'makespan': 1888}, {'dataset': 'la36', 'makespan': 1268}, {'dataset': 'la37', 'makespan': 1397}, {'dataset': 'la38', 'makespan': 1196}, {'dataset': 'la39', 'makespan': 1233}, {'dataset': 'la40', 'makespan': 1222} ]

lp_data_string = """
Instance Name   | Makespan (Heuristic) | LP Lower Bound  | Optimal Makespan      
--------------------------------------------------------------------------
la01            | 861.00               | 413.00          | 666                   
la02            | 831.00               | 394.00          | 655                   
la03            | 931.00               | 349.00          | 597                   
la04            | 877.00               | 369.00          | 590                   
la05            | 683.00               | 380.00          | 593                   
la06            | 1303.00              | 413.00          | 926                   
la07            | 1124.00              | 376.00          | 890                   
la08            | 1333.00              | 369.00          | 863                   
la09            | 1119.00              | 382.00          | 951                   
la10            | 1116.00              | 443.00          | 958                   
la11            | 1479.00              | 413.00          | 1222                  
la12            | 1483.00              | 408.00          | 1039                  
la13            | 1452.00              | 382.00          | 1150                  
la14            | 1551.00              | 443.00          | 1292                  
la15            | 1610.00              | 378.00          | 1207                  
la16            | 1362.00              | 717.00          | 945                   
la17            | 1071.00              | 646.00          | 784                   
la18            | 1111.00              | 663.00          | 848                   
la19            | 1160.00              | 617.00          | 842                   
la20            | 1222.00              | 756.00          | 902                   
la21            | 1542.00              | 717.00          | 1046                  
la22            | 1631.00              | 619.00          | 927                   
la23            | 1440.00              | 640.00          | 1032                  
la24            | 1677.00              | 704.00          | 935                   
la25            | 1666.00              | 723.00          | 977                   
la26            | 2025.00              | 717.00          | 1218                  
la27            | 1812.00              | 686.00          | 1235                  
la28            | 1976.00              | 756.00          | 1216                  
la29            | 1787.00              | 723.00          | 1152                  
la30            | 2239.00              | 726.00          | 1355                  
la31            | 2754.00              | 717.00          | 1784                  
la32            | 2527.00              | 756.00          | 1850                  
la33            | 2655.00              | 723.00          | 1719                  
la34            | 2210.00              | 656.00          | 1721                  
la35            | 2815.00              | 647.00          | 1888                  
la36            | 1723.00              | 948.00          | 1268                  
la37            | 2055.00              | 986.00          | 1397                  
la38            | 1810.00              | 943.00          | 1196                  
la39            | 1890.00              | 922.00          | 1233                  
la40            | 1822.00              | 955.00          | 1222                  
"""

instances = [f'LA{i:02d}' for i in range(1, 41)]

lines = [line.strip() for line in lp_data_string.strip().split('\n')]
data_lines = [line for line in lines if not line.startswith('---') and line]

parsed_data = []
header = [h.strip() for h in data_lines[0].split('|')]
for line in data_lines[1:]:
    values = [v.strip() for v in line.split('|')]
    parsed_data.append(dict(zip(header, values)))


lp_df = pd.DataFrame(parsed_data)
lp_df['Instance Name'] = lp_df['Instance Name'].str.upper()
lp_df = lp_df.set_index('Instance Name')
lp_df['Makespan (Heuristic)'] = pd.to_numeric(lp_df['Makespan (Heuristic)'])
lp_df['Optimal Makespan'] = pd.to_numeric(lp_df['Optimal Makespan'])

cp_sat_data = {item['dataset'].upper(): item['makespan'] for item in CP_SAT_data_list}

df = pd.DataFrame(index=instances)
df['Cutting Plane'] = df.index.map(cutting_plane_results)
df['SA'] = df.index.map(SA_Results)
df['CP-SAT'] = df.index.map(cp_sat_data)
df['LP Heuristic'] = df.index.map(lp_df['Makespan (Heuristic)'])
df['Optimal'] = df.index.map(lp_df['Optimal Makespan'])


plot_title = 'Makespan Comparison Across LA Instances'
xlabel = 'Instance'
ylabel = 'Makespan'

legend_labels = {
    'Cutting Plane': 'Cutting Plane',
    'SA': 'Simulated Annealing (SA)',
    'LP Heuristic': 'LP Heuristic Rounding',
    'CP-SAT': 'CP-SAT',
    'Optimal': 'Optimal Solution'
}

plt.figure(figsize=(18, 8))

plt.plot(df.index, df['Cutting Plane'], marker='s', linestyle='--', label=legend_labels['Cutting Plane'])
# plt.plot(df.index, df['SA'], marker='o', linestyle=':', label=legend_labels['SA'])
# plt.plot(df.index, df['LP Heuristic'], marker='^', linestyle='-.', label=legend_labels['LP Heuristic'])

# plt.plot(df.index, df['CP-SAT'], marker='x', linestyle='--', label=legend_labels['CP-SAT']) # <--- 新增
plt.plot(df.index, df['Optimal'], marker='*', linestyle='-', linewidth=2, label=legend_labels['Optimal']) # <--- 修改样式

plt.title(plot_title, fontsize=16)
plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)

plt.xticks(rotation=90, fontsize=9)
plt.yticks(fontsize=10)

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()


output_filename = 'makespan_comparison_plot_wo3.png'
plt.savefig(output_filename)

print(f"图表已保存为 {output_filename}")