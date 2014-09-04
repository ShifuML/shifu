/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.plugin.spark.stats.unitstates;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.Array;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.SerializedCategoricalValueObject;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

public class DiscreteBinningUnitState implements UnitState {

    private static final long serialVersionUID = 1L;
    private Map<String, Integer> categoryHistNeg;
    private Map<String, Integer> categoryHistPos;
    private Map<String, Double> categoryWeightPos;
    private Map<String, Double> categoryWeightNeg;
    private Set<String> categorySet;
    
    public DiscreteBinningUnitState() {
        this.categoryHistNeg= new HashMap<String, Integer>();
        this.categoryHistPos= new HashMap<String, Integer>();
        this.categoryWeightNeg= new HashMap<String, Double>();
        this.categoryWeightPos= new HashMap<String, Double>();
        this.categorySet = new HashSet<String>();
    }
    
    public UnitState getNewBlank() {
        return new DiscreteBinningUnitState();
    }

    public void merge(UnitState state) throws Exception {
        if(!(state instanceof DiscreteBinningUnitState))
            throw new Exception("Expected DiscreteBinningState, got " + state.getClass().toString());
        // merge all maps
        DiscreteBinningUnitState newState= (DiscreteBinningUnitState) state;
        for(String key: newState.getCategoryHistNeg().keySet())
            incMapCnt(this.categoryHistNeg, key, newState.getCategoryHistNeg().get(key));
        for(String key: newState.getCategoryHistPos().keySet())
            incMapCnt(this.categoryHistPos, key, newState.getCategoryHistPos().get(key));
        for(String key: newState.getCategoryWeightNeg().keySet())
            incMapCnt(this.categoryWeightNeg, key, newState.getCategoryWeightNeg().get(key));
        for(String key: newState.getCategoryWeightPos().keySet())
            incMapCnt(this.categoryWeightPos, key, newState.getCategoryWeightPos().get(key));
        this.categorySet.addAll(newState.getCategorySet());
        
    }
    
    private void incMapCnt(Map<String, Integer> map, String key, int by) {
        int cnt = map.containsKey(key) ? map.get(key) : 0;
        map.put(key, cnt + by);
    }
    
    private void incMapCnt(Map<String, Double> map, String key, Double by) {
        double cnt = map.containsKey(key) ? map.get(key) : 0;
        map.put(key, cnt + by);
    }
    
    public void addData(Object value) {
        SerializedCategoricalValueObject cvo= (SerializedCategoricalValueObject) value;
        String category= cvo.getValue();
        categorySet.add(category);

        if (cvo.getIsPositive()) {
            incMapCnt(categoryHistPos, category, 1);
            incMapCnt(categoryWeightPos, category, cvo.getWeight());
        } else {
            incMapCnt(categoryHistNeg, category, 1);
            incMapCnt(categoryWeightNeg, category, cvo.getWeight());
        }        
    }

    
    public void populateUnivariateStats(UnivariateStats univariateStats,
            Params params) {
        DiscrStats discrStats= univariateStats.getDiscrStats();
        
        if(discrStats== null)
            discrStats= new DiscrStats();

        Map<String, Double> categoryPosRateMap = new HashMap<String, Double>();
        List<String> binCategory = new ArrayList<String>();
        List<Integer> binCountNeg = new ArrayList<Integer>();
        List<Integer> binCountPos = new ArrayList<Integer>();
        List<Integer> binCountAll = new ArrayList<Integer>();
        List<Double> binPosRate = new ArrayList<Double>();
        List<Double> binWeightedNeg = new ArrayList<Double>();
        List<Double> binWeightedPos = new ArrayList<Double>();
        List<Double> binWeightedAll = new ArrayList<Double>();

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

            Double weightedNeg = categoryWeightNeg.containsKey(key) ? categoryWeightNeg.get(key) : 0.0;
            binWeightedNeg.add(weightedNeg);

            Double weightedPos = categoryWeightPos.containsKey(key) ? categoryWeightPos.get(key) : 0.0;
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

    public Map<String, Integer> getCategoryHistNeg() {
        return categoryHistNeg;
    }

    public Map<String, Integer> getCategoryHistPos() {
        return categoryHistPos;
    }

    public Map<String, Double> getCategoryWeightPos() {
        return categoryWeightPos;
    }

    public Map<String, Double> getCategoryWeightNeg() {
        return categoryWeightNeg;
    }

    public Set<String> getCategorySet() {
        return categorySet;
    }

    private static class MapComparator implements Comparator<String> {
        Map<String, Double> base;

        public MapComparator(Map<String, Double> base) {
            this.base = base;
        }

        public int compare(String a, String b) {
            if(a.equals(b))
                return 0;
            
            int isEquals= base.get(a).compareTo(base.get(b)); 
            // prevent two doubles from being equal to each other to preserve consistency with .equals()
            return (isEquals==0)?1:isEquals;
        }
    }

}
