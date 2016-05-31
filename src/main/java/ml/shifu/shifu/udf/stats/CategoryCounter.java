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

    public CategoryCounter(List<String> categories) {

        for (String category : categories) {
            categoryMap.put(category, 0);
        }

        this.categories = categories;
    }

    @Override
    public void addData(String val) {
        if (categoryMap.containsKey(val)) {
            categoryMap.put(val, categoryMap.get(val) + 1);
        }
    }

    @Override
    public List<Integer> getCounter() {
        List<Integer> counters = new ArrayList<Integer>();

        for (int i = 0 ; i < categories.size(); i ++) {
            counters.add(categoryMap.get(categories.get(i)));
        }

        return counters;
    }
}
