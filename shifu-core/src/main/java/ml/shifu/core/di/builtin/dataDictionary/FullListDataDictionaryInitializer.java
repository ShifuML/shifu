package ml.shifu.core.di.builtin.dataDictionary;

import ml.shifu.core.di.spi.DataDictionaryInitializer;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FullListDataDictionaryInitializer implements DataDictionaryInitializer {

    static Logger log = LoggerFactory.getLogger(FullListDataDictionaryInitializer.class);

    public DataDictionary init(RequestObject req) {

        Params params = req.getGlobalParams();

        log.info("Initializing DataDictionary: " + FullListDataDictionaryInitializer.class);

        DataDictionary dict = new DataDictionary();

        String filePath = (String) params.get("pathFields");

        Scanner scanner = null;

        try {

            scanner = new Scanner(new BufferedReader(new FileReader(filePath)));

            List<DataField> fields = new ArrayList<DataField>();

            while (scanner.hasNextLine()) {
                String[] parts = scanner.nextLine().split(",");
                DataField field = new DataField();
                field.setName(FieldName.create(parts[0].trim()));
                field.setOptype(OpType.valueOf(parts[1].trim().toUpperCase()));
                if (parts.length >= 3) {
                    field.setDataType(DataType.valueOf(parts[2].trim().toUpperCase()));
                } else {
                    if (field.getOptype().equals(OpType.CONTINUOUS)) {
                        field.setDataType(DataType.DOUBLE);
                    } else {
                        field.setDataType(DataType.STRING);
                    }
                }
                fields.add(field);
            }

            dict.withDataFields(fields);
            dict.setNumberOfFields(fields.size());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to initialize DataDictionary.");
        } finally {
            if (scanner != null) {
                scanner.close();
            }
        }

        return dict;

    }
}
