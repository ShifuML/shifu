package ml.shifu.shifu.udf.stats;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * counter for the categorical val
 */
public class CategoryCounter extends Counter {

    private Map<String, Integer> categoryMap = new TreeMap<String, Integer>();
    private List<String> categories;
    private Integer missCounter;

    public CategoryCounter(List<String> categories) {

        for (String category : categories) {
            categoryMap.put(category, 0);
        }

        this.categories = categories;
        this.missCounter = 0;
    }

    @Override
    public void addData(Object val) {
        if (val != null) {
            String sVal = val.toString();
            if (categoryMap.containsKey(sVal)) {
                categoryMap.put(sVal, categoryMap.get(sVal) + 1);
            } else {
                missCounter++;
            }
        } else {
            missCounter++;
        }
    }

    @Override
    public List<Integer> getCounter() {
        List<Integer> counters = new ArrayList<Integer>();

        for (int i = 0 ; i < categories.size(); i ++) {
            counters.add(categoryMap.get(categories.get(i)));
        }

        counters.add(missCounter);

        return counters;
    }
}
