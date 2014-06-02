package ml.shifu.shifu.util;


import java.util.ArrayList;
import java.util.List;

public class LocalDataTransposer {

    public static List<List<String>> transpose(List<List<String>> rows) {

        List<List<String>> columns = new ArrayList<List<String>>();

        int size = rows.get(0).size();

        for (int i = 0; i < size; i++) {
            columns.add(new ArrayList<String>());
        }

        for (List<String> row : rows) {
            for (int i = 0; i < size; i++) {
                columns.get(i).add(row.get(i));
            }
        }

        return columns;
    }

}
