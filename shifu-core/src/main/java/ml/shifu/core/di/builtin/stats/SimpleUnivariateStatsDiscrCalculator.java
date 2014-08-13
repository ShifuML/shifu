package ml.shifu.core.di.builtin.stats;

import ml.shifu.core.util.Params;
import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.Array;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.UnivariateStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class SimpleUnivariateStatsDiscrCalculator {
    private Logger log = LoggerFactory.getLogger(SimpleUnivariateStatsDiscrCalculator.class);

    public void calculate(UnivariateStats univariateStats, List<?> values, Params params) {

        DiscrStats discrStats = new DiscrStats();

        Map<String, Integer> categoryCount = new HashMap<String, Integer>();


        Set<String> categorySet = new HashSet<String>();


        for (Object value : values) {

            String category = value.toString();
            categorySet.add(category);


            incMapCnt(categoryCount, category);

        }


        Array countArray = new Array();
        countArray.setType(Array.Type.INT);
        countArray.setN(categorySet.size());
        countArray.setValue(StringUtils.join(categoryCount.values(), " "));

        Array stringArray = new Array();
        stringArray.setType(Array.Type.STRING);
        stringArray.setN(categorySet.size());
        stringArray.setValue(StringUtils.join(categoryCount.keySet(), " "));

        discrStats.withArrays(countArray, stringArray);


        univariateStats.setDiscrStats(discrStats);
    }


    private void incMapCnt(Map<String, Integer> map, String key) {
        int cnt = map.containsKey(key) ? map.get(key) : 0;
        map.put(key, cnt + 1);
    }


}
