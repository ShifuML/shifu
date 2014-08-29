package ml.shifu.plugin.createdd;

import com.google.common.base.Splitter;
import ml.shifu.core.di.spi.PMMLDataDictionaryCreator;
import ml.shifu.core.util.PMMLUtils;
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
        String delimiter = (String) params.get("delimiter", ",");
        String categorical = (String) params.get("categoricalColumnNameFile","");
        String ordinal = (String) params.get("ordinalColumnNameFile","");
        String continious = (String) params.get("continiousColumnNameFile","");
        Map<String, String> categoricalMap = new HashMap<String, String>();
        Map<String, String> ordinalMap = new HashMap<String,String>();
        Map<String, String> continiousMap = new HashMap<String,String>();
        
        String header;
        Scanner scanner = null;
        String line;

        try {
            scanner = new Scanner(new BufferedReader(new FileReader(categorical)));
            while(scanner.hasNextLine()) {
                line = scanner.nextLine();
                String list[] = line.split(delimiter);
                if(list.length > 1) categoricalMap.put(list[0],list[1]);
                else categoricalMap.put(list[0],"string");
            }
 
            scanner = new Scanner(new BufferedReader(new FileReader(ordinal)));
            while(scanner.hasNextLine()) {
                line = scanner.nextLine();
                String list[] = line.split(delimiter);
                if(list.length > 1) categoricalMap.put(list[0],list[1]);
                else ordinalMap.put(list[0],"string");
            }
 
            scanner = new Scanner(new BufferedReader(new FileReader(continious)));
            while(scanner.hasNextLine()) {
                line = scanner.nextLine();
                String list[] = line.split(delimiter);
                if(list.length > 1) categoricalMap.put(list[0],list[1]);
                else continiousMap.put(list[0],"string");
            }
 
            scanner = new Scanner(new BufferedReader(new FileReader(pathCSV)));
            header = scanner.nextLine();

            List<DataField> fields = new ArrayList<DataField>();
            for (String fieldNameString : Splitter.on(delimiter).split(header)) {
                DataField field = new DataField();
                field.setName(FieldName.create(fieldNameString));

                //Params fieldParams = params.getFieldConfig(fieldNameString).getParams();
                //field.setOptype(PMMLUtils.getOpTypeFromParams(fieldParams));
                //field.setDataType(PMMLUtils.getDataTypeFromParams(fieldParams));
                if(ordinalMap.containsKey(fieldNameString)) {
                    field.setOptype(OpType.ORDINAL);
                    field.setDataType(DataType.valueOf((String)ordinalMap.get("fieldNameString")));
                }
                else if(continiousMap.containsKey(fieldNameString)) {
                    field.setOptype(OpType.CONTINUOUS);
                    field.setDataType(DataType.valueOf((String)continiousMap.get("fieldNameString")));
                }
                else if(categoricalMap.containsKey(fieldNameString)) {
                    field.setOptype(OpType.CATEGORICAL);
                    field.setDataType(DataType.valueOf((String)continiousMap.get("fieldNameString")));
                }
                else {
                    field.setOptype(OpType.CONTINUOUS);
                }

                fields.add(field);
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
}
