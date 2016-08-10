/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.gs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import ml.shifu.shifu.core.processor.TrainModelProcessor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Grid search supports for kinds of parameters set by List: [..., ..., ...].
 * 
 * <p>
 * {@link #getParams(int)} list real params can be set in train part.
 * 
 * <p>
 * In {@link TrainModelProcessor} there is logic that to process all kind of hyper params.
 * 
 * <p>
 * Only distributed model supports grid search functions.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class GridSearch {

    protected static final Logger LOG = LoggerFactory.getLogger(GridSearch.class);

    private final Map<String, Object> rawParams;

    private List<Map<String, Object>> flattenParams = new ArrayList<Map<String, Object>>();

    private int hyperParamCount;

    private int flattenParamsCount;

    public GridSearch(Map<String, Object> rawParams) {
        assert rawParams != null;
        this.rawParams = rawParams;
        parseParams(this.rawParams);
    }

    @SuppressWarnings("rawtypes")
    private void parseParams(Map<String, Object> params) {
        SortedMap<String, Object> sortedMap = new TreeMap<String, Object>(params);
        LOG.debug(sortedMap.toString());
        List<Integer> hyperParamCntList = new ArrayList<Integer>();
        Map<String, Object> normalParams = new HashMap<String, Object>();
        List<Tuple> hyperParams = new ArrayList<GridSearch.Tuple>();

        for(Entry<String, Object> entry: sortedMap.entrySet()) {
            if(entry.getKey().equals("ActivationFunc") || entry.getKey().equals("NumHiddenNodes")) {
                if(entry.getValue() instanceof List) {
                    if(((List) (entry.getValue())).size() > 0 && ((List) (entry.getValue())).get(0) instanceof List) {
                        this.hyperParamCount += 1;
                        hyperParams.add(new Tuple(entry.getKey(), entry.getValue()));
                        hyperParamCntList.add(((List) entry.getValue()).size());
                    } else {
                        normalParams.put(entry.getKey(), entry.getValue());
                    }
                }
                continue;
            } else if(entry.getValue() instanceof List) {
                this.hyperParamCount += 1;
                hyperParams.add(new Tuple(entry.getKey(), entry.getValue()));
                hyperParamCntList.add(((List) entry.getValue()).size());
            } else {
                normalParams.put(entry.getKey(), entry.getValue());
            }
        }

        // TODO parameter validation

        if(hasHyperParam()) {
            this.flattenParamsCount = 1;
            for(Integer cnt: hyperParamCntList) {
                this.flattenParamsCount *= cnt;
            }
            // construct flatten params map
            for(int i = 0; i < this.flattenParamsCount; i++) {
                Map<String, Object> map = new HashMap<String, Object>();
                int amplifier = 1;
                for(int j = hyperParamCntList.size() - 1; j >= 0; j--) {
                    int currParamCnt = hyperParamCntList.get(j);
                    Tuple tuple = hyperParams.get(j);
                    Object value = ((List) (tuple.value)).get(i / amplifier % currParamCnt);
                    map.put(tuple.key, value);
                    amplifier *= currParamCnt;
                }
                for(Entry<String, Object> entry: normalParams.entrySet()) {
                    map.put(entry.getKey(), entry.getValue());
                }
                this.flattenParams.add(map);
            }
        }
    }

    public int hyperParamCount() {
        return this.hyperParamCount;
    }

    public Map<String, Object> getParams(int i) {
        return this.flattenParams.get(i);
    }

    public List<Map<String, Object>> getFlattenParams() {
        return this.flattenParams;
    }

    public boolean hasHyperParam() {
        return this.hyperParamCount > 0;
    }

    public boolean isGridSearchMode() {
        return this.hyperParamCount > 0;
    }

    private static class Tuple {

        public Tuple(String key, Object value) {
            this.key = key;
            this.value = value;
        }

        public String key;
        public Object value;
    }

    // public static void main(String[] args) {
    // Map<String, Object> map = new HashMap<String, Object>();
    // map.put("TreeNum", Arrays.asList(5, 10, 11));
    // map.put("Loss", Arrays.asList("sqaured", "log"));
    // map.put("NumHiddenNodes", Arrays.asList(Arrays.asList(10), Arrays.asList(20)));
    // map.put("ActivationFunc", Arrays.asList("tanh"));
    // map.put("MaxDepth", Arrays.asList(8, 10, 12));
    // map.put("LearningRate", 0.1);
    // map.put("Impurity", "entropy");
    // GridSearch gs = new GridSearch(map);
    // System.out.println(gs.flattenParams);
    // }
}
