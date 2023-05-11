data = LOAD '$path_raw_data' USING $pig_data_load;
data = FILTER data BY a == 1;