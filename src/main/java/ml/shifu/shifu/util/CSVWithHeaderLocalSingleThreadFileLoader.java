package ml.shifu.shifu.util;

import ml.shifu.shifu.di.spi.SingleThreadFileLoader;
import org.apache.commons.lang.ArrayUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class CSVWithHeaderLocalSingleThreadFileLoader implements SingleThreadFileLoader {

    private String delimiter = ",";

    public List<List<String>> load(String filePath) {
        Scanner scanner = null;
        List<List<String>> rows = new ArrayList<List<String>>();
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(filePath)));

            // Discard Header
            scanner.nextLine();

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();

                if (line == null || line.length() == 0) {
                    continue;
                }

                List<String> fields = Arrays.asList(line.split(delimiter));
                rows.add(fields);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Cannot load file");
        } finally {
            if (scanner == null) {
                scanner.close();
            }
        }

        return rows;
    }

    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

}
