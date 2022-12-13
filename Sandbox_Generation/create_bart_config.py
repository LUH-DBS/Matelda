import math
from multiprocessing import parent_process
from lxml import etree
import os

from matplotlib.pyplot import table


def generate_error_definition(table_name, columns, fd_list):
    table_error_definiotion = "DCs: \n"
    for i in range(1, len(fd_list)+1):
        determinant = fd_list[i-1][0]
        dependant = fd_list[i-1][1]

        table_error_definiotion += f'''
            e{i}: {table_name}({", ".join(f"{col}: ${col}1" for col in columns)}),
            {table_name}({", ".join(f"{col}: ${col}2" for col in columns)}),
                ${determinant}1 == ${determinant}2, ${dependant}1 != ${dependant}2 -> #fail. \n
        '''

    return table_error_definiotion

def set_source_config(xml_root, input_file, table_name):
    input_obj_src = xml_root.xpath("//source/import/input")[0]
    input_obj_src.text = os.path.abspath(input_file)
    input_obj_src.set('table', table_name)
    return xml_root

def set_target_config(xml_root, input_file, table_name):
    db_conf_obj = xml_root.xpath("//target/access-configuration/uri")[0]
    db_conf_obj.text = 'jdbc:postgresql://localhost:5432/' + table_name
    input_obj_trgt = xml_root.xpath("//target/import/input")[0]
    input_obj_trgt.text = os.path.abspath(input_file)
    input_obj_trgt.set('table', table_name)
    return xml_root

def set_outliers_config(xml_root, table_name, outlier_error_cols, outlier_errors_percentage):
    outlier_error_obj = xml_root.xpath("//outlierErrors/tables/table")[0]
    outlier_error_obj.set('name', table_name)

    outlier_error_obj_attr= xml_root.find("//outlierErrors/tables/table/attributes")
    for col in outlier_error_cols:
        out_atrr = etree.SubElement(outlier_error_obj_attr, "atrribute")
        out_atrr.set("percentage", str(math.floor(int(outlier_errors_percentage)/len(outlier_error_cols))))
        out_atrr.set("detectable", "true")
        out_atrr.text = col
    return xml_root

def set_typos_config(xml_root, table_name, typo_percentage, typo_cols):
    typo_error_obj_table = xml_root.xpath("//randomErrors/tables/table")[0]
    typo_error_obj_table.set('name', table_name)

    typo_error_obj_percentage = xml_root.xpath("//randomErrors/tables/table/percentage")[0]
    typo_error_obj_percentage.text = str(typo_percentage)

    typo_error_obj_attr= xml_root.find("//randomErrors/tables/table/attributes")
    for col in typo_cols:
        typo_atrr = etree.SubElement(typo_error_obj_attr, "atrribute")
        typo_atrr.text = col
    return xml_root

def set_fds_config(xml_root, id, determinant, dependant, vio_gen_percentage):

    parent_vio_gen = xml_root.xpath("//errorPercentages/vioGenQueries")[0]
    vio_gen_obj_1 = etree.SubElement(parent_vio_gen, "vioGenQuery")
    vio_gen_obj_1.set("id", id)
    vio_gen_obj_2 = etree.SubElement(parent_vio_gen, "vioGenQuery")
    vio_gen_obj_2.set("id", id)

    vio_gen_comparison_obj_1 = etree.SubElement(vio_gen_obj_1, "comparison")
    vio_gen_comparison_obj_1.text = f'''({determinant}1 == {determinant}2)'''
    vio_gen_comparison_obj_2 = etree.SubElement(vio_gen_obj_2, "comparison")
    vio_gen_comparison_obj_2.text = f'''({dependant}1 != {dependant}2)'''

    vio_gen_percentage_obj_1 = etree.SubElement(vio_gen_obj_1, "percentage")
    vio_gen_percentage_obj_1.text = str(vio_gen_percentage / 2)
    vio_gen_percentage_obj_2 = etree.SubElement(vio_gen_obj_2, "percentage")
    vio_gen_percentage_obj_2.text = str(vio_gen_percentage / 2)

    return xml_root


def set_config(xml_root, input_file, table_columns, outlier_error_cols, outlier_errors_percentage, typo_cols, typo_percentage, fd_ratio_dict, output_dict):

    table_name = os.path.basename(input_file).replace('.csv', '')
    xml_root = set_target_config(xml_root, input_file, table_name)

    xml_root = set_outliers_config(xml_root, table_name, outlier_error_cols, outlier_errors_percentage)
    xml_root = set_typos_config(xml_root, table_name, typo_percentage, typo_cols)

    error_definition = generate_error_definition(table_name, table_columns, list(fd_ratio_dict.keys()))
    dependencies_obj = xml_root.xpath("//dependencies")[0]
    dependencies_obj.text = etree.CDATA(error_definition)

    for idx, fd in enumerate(list(fd_ratio_dict.keys())):
        xml_root = set_fds_config(xml_root, f'''e{idx+1}''', str(fd[0]), str(fd[1]), fd_ratio_dict[fd])
    
    export_dirty_db_obj = xml_root.xpath("//configuration/exportDirtyDBPath")[0] 
    export_dirty_db_obj.text = os.path.join(output_dict, table_name)
    export_dirty_file_obj = xml_root.xpath("//configuration/exportCellChangesPath")[0]
    export_dirty_file_obj. text = os.path.join(output_dict, table_name, table_name + ".csv")

    return xml_root

def create_config_file(input_file, table_columns, outlier_error_col, outlier_errors_percentage, typo_col, typo_percentage, fd_ratio_dict, output_dir):
    parser = etree.XMLParser(strip_cdata=False)
    root_tree = etree.parse('Sandbox_Generation/bart_sample_config.xml', parser=parser)

    set_config(root_tree, input_file, table_columns, outlier_error_col, outlier_errors_percentage, typo_col, typo_percentage, fd_ratio_dict, output_dir)
    config_file_path = os.path.join(output_dir, f'''bart_config_{os.path.basename(input_file).replace('.csv', '')}.xml''')
    root_tree.write(config_file_path)
    return config_file_path
