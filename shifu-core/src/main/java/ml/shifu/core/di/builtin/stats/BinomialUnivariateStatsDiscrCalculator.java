package ml.shifu.core.di.builtin.stats;

import ml.shifu.core.container.CategoricalValueObject;
import ml.shifu.core.util.PMMLUtils;
import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.Array;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.UnivariateStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class BinomialUnivariateStatsDiscrCalculator {
    private static final Logger log = LoggerFactory.getLogger(BinomialUnivariateStatsDiscrCalculator.class);

    public static void calculate(UnivariateStats univariateStats, List<CategoricalValueObject> cvoList, Map<String, Object> params) {

        DiscrStats discrStats = new DiscrStats();

        Map<String, Integer> categoryHistNeg = new HashMap<String, Integer>();
        Map<String, Integer> categoryHistPos = new HashMap<String, Integer>();
        Map<String, Double> categoryWeightedNeg = new HashMap<String, Double>();
        Map<String, Double> categoryWeightedPos = new HashMap<String, Double>();


        List<String> binCategory = new ArrayList<String>();
        List<Integer> binCountNeg = new ArrayList<Integer>();
        List<Integer> binCountPos = new ArrayList<Integer>();
        List<Integer> binCountAll = new ArrayList<Integer>();
        List<Double> binPosRate = new ArrayList<Double>();
        List<Double> binWeightedNeg = new ArrayList<Double>();
        List<Double> binWeightedPos = new ArrayList<Double>();
        List<Double> binWeightedAll = new ArrayList<Double>();

        Set<String> categorySet = new HashSet<String>();

        int voSize = cvoList.size();

        for (CategoricalValueObject vo : cvoList) {

            String category = vo.getValue();
            categorySet.add(category);

            if (vo.getIsPositive()) {
                incMapCnt(categoryHistPos, category);
                incMapWithValue(categoryWeightedPos, category, vo.getWeight());
            } else {
                incMapCnt(categoryHistNeg, category);
                incMapWithValue(categoryWeightedNeg, category, vo.getWeight());
            }
        }

        Map<String, Double> categoryPosRateMap = new HashMap<String, Double>();

        for (String key : categorySet) {
            double cnt0 = categoryHistNeg.containsKey(key) ? categoryHistNeg.get(key) : 0;
            double cnt1 = categoryHistPos.containsKey(key) ? categoryHistPos.get(key) : 0;
            double rate;
            if (cnt0 + cnt1 == 0) {
                rate = 0;
            } else {
                rate = cnt1 / (cnt0 + cnt1);
            }
            categoryPosRateMap.put(key, rate);
        }

        // Sort map
        MapComparator cmp = new MapComparator(categoryPosRateMap);
        Map<String, Double> sortedBinPosRateMap = new TreeMap<String, Double>(cmp);
        sortedBinPosRateMap.putAll(categoryPosRateMap);

        for (String key : sortedBinPosRateMap.keySet()) {
            Integer countNeg = categoryHistNeg.containsKey(key) ? categoryHistNeg.get(key) : 0;
            binCountNeg.add(countNeg);
            Integer countPos = categoryHistPos.containsKey(key) ? categoryHistPos.get(key) : 0;
            binCountPos.add(countPos);

            binCountAll.add(countNeg + countPos);

            Double weightedNeg = categoryWeightedNeg.containsKey(key) ? categoryWeightedNeg.get(key) : 0.0;
            binWeightedNeg.add(weightedNeg);

            Double weightedPos = categoryWeightedPos.containsKey(key) ? categoryWeightedPos.get(key) : 0.0;
            binWeightedPos.add(weightedPos);

            binWeightedAll.add(weightedNeg + weightedPos);

            // use zero, the average score is calculate in post-process

            binCategory.add(key);
            binPosRate.add(sortedBinPosRateMap.get(key));
        }

        Array countArray = new Array();
        countArray.setType(Array.Type.INT);
        countArray.setN(binCountAll.size());
        countArray.setValue(StringUtils.join(binCountAll, " "));

        discrStats.withArrays(countArray);

        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", binCountPos.toString());
        extensionMap.put("BinCountNeg", binCountNeg.toString());
        extensionMap.put("BinWeightedCountPos", binWeightedPos.toString());
        extensionMap.put("BinWeightedCountNeg", binWeightedNeg.toString());
        extensionMap.put("BinPosRate", binPosRate.toString());

        discrStats.withExtensions(PMMLUtils.createExtensions(extensionMap));

        univariateStats.setDiscrStats(discrStats);
                
    }


    private static void incMapCnt(Map<String, Integer> map, String key) {
        int cnt = map.containsKey(key) ? map.get(key) : 0;
        map.put(key, cnt + 1);
    }

    private static void incMapWithValue(Map<String, Double> map, String key, Double value) {
        double num = map.containsKey(key) ? map.get(key) : 0.0;
        map.put(key, num + value);
    }

    private static class MapComparator implements Comparator<String> {
        Map<String, Double> base;

        public MapComparator(Map<String, Double> base) {
            this.base = base;
        }

        public int compare(String a, String b) {
            return base.get(a).compareTo(base.get(b));
        }
    }
}
