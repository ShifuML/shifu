package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.di.spi.DataDictionaryInitializer;
import ml.shifu.shifu.pmml.obj.DataDictionary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class CSVDataDictionaryInitializer implements DataDictionaryInitializer {

    static Logger log = LoggerFactory.getLogger(CSVDataDictionaryInitializer.class);

    public DataDictionary init(Map<String, Object> params) {
        DataDictionary dict = new DataDictionary();

        String filePath;
        String delimiter = ",";

        if (params.containsKey("filePath")) {
            filePath = (String) params.get("filePath");
            log.info("filePath: " + filePath);
        } else {
            throw new RuntimeException("Required Param Missing: filePath");
        }

        if (params.containsKey("delimiter")) {
            delimiter = (String) params.get("delimiter");
            log.info("Using user specified delimiter: " + delimiter);
        } else {
            log.info("Using default delimiter: ,");
        }

        String header;
        Scanner scanner = null;
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(filePath)));
            header = scanner.nextLine();


            List<DataDictionary.DataField> fields = new ArrayList<DataDictionary.DataField>();
            for (String fieldName : header.split(delimiter)) {
                DataDictionary.DataField field = new DataDictionary.DataField();
                field.setName(fieldName);
                fields.add(field);
            }

            dict.setDataFields(fields);
            dict.setNumberOfFields(fields.size());
        } catch (Exception e) {
            throw new RuntimeException("Cannot load file.");
        } finally {
            if (scanner != null) {
                scanner.close();
            }
        }



        return dict;

    }
}
