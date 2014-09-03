## shifu-plugin-createdd
## Plugin to Create Data Dictionary

Plugin to be used in place of ml.shifu.core.di.builtin.datadictionary.CSVPMMLDataDictionaryCreator

Requires file with specific information for each column name.
Default file name is: columns.txt
    
columns.txt is a 3 column csv file: fieldName,DataType,OpType

For example:  diagnosis,string,continuous

The default values for each column is string,continuous. 
If a field name is not listed in columns.txt, that field will receive default values.