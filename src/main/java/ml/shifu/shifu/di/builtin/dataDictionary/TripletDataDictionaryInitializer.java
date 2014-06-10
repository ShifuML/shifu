package ml.shifu.shifu.di.builtin.dataDictionary;

import ml.shifu.shifu.di.spi.DataDictionaryInitializer;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class TripletDataDictionaryInitializer implements DataDictionaryInitializer {

    static Logger log = LoggerFactory.getLogger(TripletDataDictionaryInitializer.class);

    public DataDictionary init(RequestObject req) {

        log.info("Initializing DataDictionary: " + TripletDataDictionaryInitializer.class);

        Params params = req.getGlobalParams();

        DataDictionary dict = new DataDictionary();

        String filePath = (String) params.get("filePath");

        Scanner scanner = null;

        try {

            scanner = new Scanner(new BufferedReader(new FileReader(filePath)));

            List<DataField> fields = new ArrayList<DataField>();

            while(scanner.hasNextLine()) {
                String[] parts = scanner.nextLine().split(",");
                DataField field = new DataField();
                field.setName(FieldName.create(parts[0].trim()));
                field.setOptype(OpType.valueOf(parts[1].trim().toUpperCase()));
                field.setDataType(DataType.valueOf(parts[2].trim().toUpperCase()));
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
