package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.di.spi.DataDictionaryTypeSetter;
import ml.shifu.shifu.pmml.obj.DataDictionary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class TripletDataDictionaryTypeSetter implements DataDictionaryTypeSetter {

    static Logger log = LoggerFactory.getLogger(TripletDataDictionaryTypeSetter.class);

    public void setType(DataDictionary dict, Map<String, Object> params) {

        String typeTripletFilePath;

        if (params.containsKey("typeTripletFilePath")) {

            typeTripletFilePath = (String) params.get("typeTripletFilePath");
            log.info("TypeTripletFilePath: " + typeTripletFilePath);
        } else {
            throw new RuntimeException("Required Param Missing: typeTripletFilePath");
        }

        Map<String, DataDictionary.DataField> dataFieldMap = new HashMap<String, DataDictionary.DataField>();
        for (DataDictionary.DataField field : dict.getDataFields()) {
            dataFieldMap.put(field.getName(), field);
        }

        Scanner scanner = null;

        try {
            scanner = new Scanner(new BufferedReader(new FileReader(typeTripletFilePath)));

            while(scanner.hasNextLine()) {
                String[] triplet = scanner.nextLine().split(",");
                if (dataFieldMap.containsKey(triplet[0])) {
                    DataDictionary.DataField field = dataFieldMap.get(triplet[0].trim());
                    field.setOptype(triplet[1].trim());
                    field.setDataType(triplet[2].trim());
                } else {
                    log.info("Field Not Found:" + triplet[0] +  ", Ignore");
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}

