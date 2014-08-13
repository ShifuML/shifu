package ml.shifu.core.util;

import com.google.common.base.Splitter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LocalDataUtils {

    private static Logger log = LoggerFactory.getLogger(LocalDataUtils.class);

    public static List<String> loadHeader(String path, String delimiter) {
        Scanner scanner = null;
        List<String> header = new ArrayList<String>();
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(path)));
            String line = scanner.nextLine();

            for (String s : Splitter.on(delimiter).split(line)) {
                header.add(s);
            }
        } catch (Exception e) {
            log.error(e.toString());
            throw new RuntimeException("Cannot load file");
        } finally {
            if (scanner != null) {
                scanner.close();
            }
        }
        return header;

    }

    public static List<List<Object>> loadData(String path, String delimiter) {
        Scanner scanner = null;
        List<List<Object>> rows = new ArrayList<List<Object>>();
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(path)));
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

}
