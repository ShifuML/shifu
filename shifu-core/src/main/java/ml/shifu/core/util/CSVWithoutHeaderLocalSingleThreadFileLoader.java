package ml.shifu.core.util;

import com.google.common.base.Splitter;
import ml.shifu.core.di.spi.SingleThreadFileLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CSVWithoutHeaderLocalSingleThreadFileLoader implements SingleThreadFileLoader {

    private static Logger log = LoggerFactory.getLogger(CSVWithoutHeaderLocalSingleThreadFileLoader.class);
    private String delimiter = ",";

    public List<List<Object>> load(String filePath) {
        Scanner scanner = null;
        List<List<Object>> rows = new ArrayList<List<Object>>();
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(filePath)));


            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();

                if (line == null || line.length() == 0) {
                    continue;
                }

                List<Object> fields = new ArrayList<Object>();
                for (String s : Splitter.on(delimiter).split(line)) {
                    fields.add(s);
                }
                rows.add(fields);
            }
        } catch (Exception e) {
            log.error(e.toString());
            throw new RuntimeException("Cannot load file");
        } finally {
            if (scanner != null) {
                scanner.close();
            }
        }

        return rows;
    }

    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

}
