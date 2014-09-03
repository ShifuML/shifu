package ml.shifu.plugin.createdd;

import com.google.common.base.Splitter;
import ml.shifu.core.di.spi.PMMLDataDictionaryCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class CSVPMMLDataDictionaryCreator implements PMMLDataDictionaryCreator {

    private static Logger log = LoggerFactory.getLogger(CSVPMMLDataDictionaryCreator.class);

    public DataDictionary create(Params params) {
        DataDictionary dict = new DataDictionary();

        String pathCSV = (String) params.get("pathCSV");
        String csvDelimiter = (String) params.get("csvDelimiter", ",");
        String nameFileDelimiter = (String) params.get("nameFileelimiter", ",");
        String columns = (String) params.get("columnNameFile","");
        Map<String, DataField> dataMap = new HashMap<String, DataField>();
        
        String header;
        Scanner scanner = null;
        String line;

        try {
        
            log.info("Loading "+columns);
            scanner = new Scanner(new BufferedReader(new FileReader(columns)));
            while(scanner.hasNextLine()) {
                line = scanner.nextLine();
                if(line.startsWith("##")) continue;
                String list[] = line.split(nameFileDelimiter);
                dataMap.put(list[0], getDataField(list));
            }
            scanner.close();
            
            scanner = new Scanner(new BufferedReader(new FileReader(pathCSV)));
            header = scanner.nextLine();
            log.info("Loaded "+pathCSV);
            List<DataField> fields = new ArrayList<DataField>();
            for (String fieldNameString : Splitter.on(csvDelimiter).split(header)) {
                
                if(dataMap.containsKey(fieldNameString)) {
                    fields.add(dataMap.get(fieldNameString));
                }
                else {
                    DataField field = new DataField();
                    field.setName(FieldName.create(fieldNameString));
                    field.setOptype(OpType.CONTINUOUS);
                    field.setDataType(DataType.STRING);
                    fields.add(field);                    
                }
                    

            }
            
            dict.withDataFields(fields);
            dict.setNumberOfFields(fields.size());
        } catch (Exception e) {
            log.error(e.toString());
            throw new RuntimeException("Cannot load file.");
        } finally {
            if (scanner != null) {
                scanner.close();
            }
        }
        

        return dict;
    }
    
    DataField getDataField(String[] list) {
        
        DataField df = new DataField();
        df.setName(FieldName.create(list[0]));
        if(list.length == 1) {
            df.setDataType(DataType.STRING);
            df.setOptype(OpType.CONTINUOUS);
        }
        else if(list.length == 2) {
            df.setDataType(DataType.valueOf(list[1].toUpperCase()));
            df.setOptype(OpType.CONTINUOUS);
        }
        else if(list.length == 3){
            df.setDataType(DataType.valueOf(list[1].toUpperCase()));
            df.setOptype(OpType.valueOf(list[2].toUpperCase()));
        }
        
        return df;
    }
}
