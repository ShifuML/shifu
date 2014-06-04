package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.di.spi.DataDictionaryInitializer;

import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class CSVDataDictionaryInitializer implements DataDictionaryInitializer {

    static Logger log = LoggerFactory.getLogger(CSVDataDictionaryInitializer.class);

    public DataDictionary init(Params params) {
        DataDictionary dict = new DataDictionary();

        String csvFilePath;
        String delimiter = ",";
        String typeFilePath = null;
         /*
        if (params.containsKey("csvFilePath")) {
            csvFilePath = (String) params.get("csvFilePath");
            log.info("csvFilePath: " + csvFilePath);
        } else {
            throw new RuntimeException("Required Param Missing: csvFilePath");
        }

        if (params.containsKey("delimiter")) {
            delimiter = (String) params.get("delimiter");
            log.info("Using user specified delimiter: " + delimiter);
        } else {
            log.info("Using default delimiter: ,");
        }

        if (params.containsKey("typeFilePath")) {
            typeFilePath = (String) params.get("typeFilePath");
        } else {
            log.warn("No TypeFile provided");
        }

        String header;
        Scanner scanner = null;
        Scanner typeFileScanner = null;
        try {

            Map<String, String> typeMap = new HashMap<String, String>();

            if (typeFilePath == null) {
                typeFileScanner = new Scanner(new BufferedReader(new FileReader(typeFilePath)));
                while (typeFileScanner.hasNextLine()) {
                    String line = typeFileScanner.nextLine();
                    String[] parts = line.split(",", 1);
                    typeMap.put(parts[0], parts[1]);
                }
            }


            scanner = new Scanner(new BufferedReader(new FileReader(csvFilePath)));
            header = scanner.nextLine();


            List<DataField> fields = new ArrayList<DataField>();
            for (String fieldNameString : header.split(delimiter)) {
                DataField field = new DataField();
                field.setName(FieldName.create(fieldNameString));

                if (typeMap.containsKey(fieldNameString)) {
                    String[] types = typeMap.get(fieldNameString).split(",");
                    field.setOptype(OpType.valueOf(types[0]));
                    field.setDataType(DataType.valueOf(types[1]));
                }


                fields.add(field);
            }

            dict.withDataFields(fields);
            dict.setNumberOfFields(fields.size());
        } catch (Exception e) {
            throw new RuntimeException("Cannot load file.");
        } finally {
            if (scanner != null) {
                scanner.close();
            }
        }
                         */
        return dict;

    }
}
