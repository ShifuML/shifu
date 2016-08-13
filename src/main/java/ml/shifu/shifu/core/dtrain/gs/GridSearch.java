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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import ml.shifu.shifu.core.processor.TrainModelProcessor;
import ml.shifu.shifu.util.Environment;

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

    /**
     * Raw parameter from ModelConfig#train#params
     */
    private final Map<String, Object> rawParams;

    /**
     * List all kinds of hyper parameters which can be used in training step. List is sorted by name + order in each
     * parameter.
     */
    private List<Map<String, Object>> flattenParams = new ArrayList<Map<String, Object>>();

    /**
     * How many hyper parameters
     */
    private int hyperParamCount;

    /**
     * How many hyper parameter composite, size of {@link #flattenParams}
     */
    private int flattenParamsCount;

    public GridSearch(Map<String, Object> rawParams) {
        assert rawParams != null;
        this.rawParams = rawParams;
        parseParams(this.rawParams);
    }

    @SuppressWarnings("rawtypes")
    private void parseParams(Map<String, Object> params) {
        // use sorted map to sort all parameters by natural order, this makes all flatten parameters sorted and fixed
        SortedMap<String, Object> sortedMap = new TreeMap<String, Object>(params);
        LOG.debug(sortedMap.toString());
        List<Integer> hyperParamCntList = new ArrayList<Integer>();
        Map<String, Object> normalParams = new HashMap<String, Object>();
        List<Tuple> hyperParams = new ArrayList<GridSearch.Tuple>();

        // stats on hyper parameters
        for(Entry<String, Object> entry: sortedMap.entrySet()) {
            if(entry.getKey().equals("ActivationFunc") || entry.getKey().equals("NumHiddenNodes")) {
                if(entry.getValue() instanceof List) {
                    if(((List) (entry.getValue())).size() > 0 && ((List) (entry.getValue())).get(0) instanceof List) {
                        // ActivationFunc and NumHiddenNodes in NN is already List, so as hyper parameter they should be
                        // list of list.
                        this.hyperParamCount += 1;
                        hyperParams.add(new Tuple(entry.getKey(), entry.getValue()));
                        hyperParamCntList.add(((List) entry.getValue()).size());
                    } else {
                        // else as normal params
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
            // compute all kinds hyper parameter composite and set into flatten Params
            // TODO, do we need a threshold like 30 since the cost of grid search is high
            this.flattenParamsCount = 1;
            for(Integer cnt: hyperParamCntList) {
                this.flattenParamsCount *= cnt;
            }
            // construct flatten params map
            for(int i = 0; i < this.flattenParamsCount; i++) {
                Map<String, Object> map = new HashMap<String, Object>();
                int amplifier = 1;
                // find hyper parameters
                for(int j = hyperParamCntList.size() - 1; j >= 0; j--) {
                    int currParamCnt = hyperParamCntList.get(j);
                    Tuple tuple = hyperParams.get(j);
                    Object value = ((List) (tuple.value)).get(i / amplifier % currParamCnt);
                    map.put(tuple.key, value);
                    amplifier *= currParamCnt;
                }
                // put normal parameters
                for(Entry<String, Object> entry: normalParams.entrySet()) {
                    map.put(entry.getKey(), entry.getValue());
                }
                this.flattenParams.add(map);
            }

            // random search if over threshold
            int threshold = Environment.getInt("shifu.gridsearch.threshold", 30);

            if(this.flattenParamsCount > threshold) {
                // set random search size is threshold
                LOG.info("Grid search numer is over threshold {}, leverage randomize search.", threshold);
                this.flattenParamsCount = threshold;
                List<Map<String, Object>> oldFlattenParams = this.flattenParams;
                this.flattenParams = new ArrayList<Map<String, Object>>(threshold);
                // just to select fixed number of elements, not random to make it can be called twice and return the
                // same result;
                int mod = oldFlattenParams.size() % threshold;
                int factor = oldFlattenParams.size() / threshold;
                for(int i = 0; i < threshold; i++) {
                    if(i > (threshold - 1 - mod)) {
                        this.flattenParams.add(oldFlattenParams.get((factor + 1) * i - (threshold - mod)));
                    } else {
                        this.flattenParams.add(oldFlattenParams.get(factor * i));
                    }
                }
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

}
