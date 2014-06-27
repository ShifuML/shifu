package ml.shifu.core.util;


import java.util.ArrayList;
import java.util.List;

public class LocalDataTransposer {

    public static List<List<Object>> transpose(List<List<Object>> rows) {

        List<List<Object>> columns = new ArrayList<List<Object>>();

        int size = rows.get(0).size();

        for (int i = 0; i < size; i++) {
            columns.add(new ArrayList<Object>());
        }

        for (List<Object> row : rows) {
            for (int i = 0; i < size; i++) {
                columns.get(i).add(row.get(i));
            }
        }

        return columns;
    }

}
