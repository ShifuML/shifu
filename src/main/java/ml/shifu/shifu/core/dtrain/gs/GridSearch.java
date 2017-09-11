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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import ml.shifu.common.Meta;
import ml.shifu.shifu.container.meta.MetaFactory;
import ml.shifu.shifu.container.meta.MetaItem;
import ml.shifu.shifu.core.processor.TrainModelProcessor;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
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

    /**
     * @param rawParams
     *            train params read from ModelConfig.json
     * @param configFileContent
     *            grid search params read from config file, raw file contents with each line as a String
     *            each line may contain one or more {param name}:{param value} group concated with ';'
     */
    public GridSearch(Map<String, Object> rawParams, List<String> configFileContent) {
        this.rawParams = rawParams;
        List<Map<String, Object>> gridParams = parseParams(configFileContent);
        if(gridParams != null && gridParams.size() > 0) {
            assert this.hasHyperParam(rawParams) == false;
            Set<String> hyperParams = new HashSet<String>();
            for(Map<String, Object> map: gridParams) {
                for(Map.Entry<String, Object> gridEntry: map.entrySet()) {
                    hyperParams.add(gridEntry.getKey());
                }
                for(Map.Entry<String, Object> kvEntry: this.rawParams.entrySet()) {
                    if(!map.containsKey(kvEntry.getKey())) {
                        map.put(kvEntry.getKey(), kvEntry.getValue());
                    }
                }
            }
            this.flattenParams = gridParams;
            this.hyperParamCount = hyperParams.size();
            this.flattenParamsCount = this.flattenParams.size();
        } else {
            // empty file content
            assert rawParams != null;
            this.hyperParamCount = 0;
            parseParams(this.rawParams);
        }
    }

    private List<Map<String, Object>> parseParams(List<String> configFileContent) {
        if(configFileContent != null) {
            Map<String, MetaItem> metaWarehouse = MetaFactory.getModelConfigMeta();
            List<Map<String, Object>> gridParams = new ArrayList<Map<String, Object>>();
            for(String configLine: configFileContent) {
                Map<String, Object> paramsMap = parseParams(metaWarehouse, configLine);
                if(paramsMap != null) {
                    gridParams.add(paramsMap);
                }
            }
            return gridParams;
        }
        return null;
    }

    private Map<String, Object> parseParams(Map<String, MetaItem> metaWarehouse, String configLine) {
        if(configLine == null || configLine.trim().equals("")) {
            return null;
        }
        String[] eles = configLine.trim().split(";");
        Map<String, Object> paramsMap = new HashMap<String, Object>();
        for(int i = 0; i < eles.length; i++) {
            int splitpos = eles[i].indexOf(':');
            if(splitpos == -1) {
                LOG.error(
                        "Error exist in train params confi file. Line content: {}. Params should be in <param name>:<param value> format, concated with ';'",
                        configLine);
                return null;
            }
            String itemKey = eles[i].substring(0, splitpos).trim();
            String itemValueStr = eles[i].substring(splitpos + 1);
            paramsMap.put(itemKey, convertItemValue(metaWarehouse, itemKey, itemValueStr));
        }
        return paramsMap;
    }

    @SuppressWarnings("rawtypes")
    private void parseParams(Map<String, Object> params) {
        // use sorted map to sort all parameters by natural order, this makes all flatten parameters sorted and fixed
        SortedMap<String, Object> sortedMap = new TreeMap<String, Object>(params);
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

    @SuppressWarnings("rawtypes")
    private boolean hasHyperParam(Map<String, Object> params) {
        for(Entry<String, Object> entry: params.entrySet()) {
            if(entry.getKey().equals("ActivationFunc") || entry.getKey().equals("NumHiddenNodes")) {
                if(entry.getValue() instanceof List) {
                    if(((List) (entry.getValue())).size() > 0 && ((List) (entry.getValue())).get(0) instanceof List) {
                        return true;
                    }
                }
            } else if(entry.getValue() instanceof List) {
                return true;
            }
        }
        return false;
    }

    private Object convertItemValue(Map<String, MetaItem> metaWarehouse, String itemKey, String itemValueStr)
            throws ShifuException {
        MetaItem itemMeta = metaWarehouse.get(getItemKeyInMeta(itemKey));
        if(itemMeta == null) {
            return null;
        }
        itemValueStr = itemValueStr.trim();
        if(itemMeta.getType().equals("text")) {
            return itemValueStr;
        } else if(itemMeta.getType().equals("number")) {
            try {
                return Double.parseDouble(itemValueStr);
            } catch (NumberFormatException e) {
                LOG.error("Train param {} should be number type, actual value got is {}", itemKey, itemValueStr);
                throw new ShifuException(ShifuErrorCode.ERROR_GRID_SEARCH_FILE_CONFIG);
            }
        } else if(itemMeta.getType().equals("boolean")) {
            return itemValueStr.equalsIgnoreCase("true");
        } else if(itemMeta.getType().equals("list")) {
            if(itemKey.equals("NumHiddenNodes") && itemMeta.getElementType().equals("number")
                    && itemValueStr.matches("\\[[0-9\\. ,]+\\]")) {
                List<Integer> itemValue = new ArrayList<Integer>();
                itemValueStr = itemValueStr.substring(1, itemValueStr.length() - 1);
                String[] splits = itemValueStr.split(",");
                for(String valueSplit: splits) {
                    itemValue.add(Integer.parseInt(valueSplit));
                }
                return itemValue;
            } else if(itemKey.equals("ActivationFunc") && itemMeta.getElementType().equals("text")
                    && itemValueStr.matches("\\[[a-zA-Z0-9 ,]+\\]")) {
                List<String> itemValue = new ArrayList<String>();
                itemValueStr = itemValueStr.substring(1, itemValueStr.length() - 1);
                String[] splits = itemValueStr.split(",");
                for(String valueSplit: splits) {
                    itemValue.add(valueSplit.trim());
                }
                return itemValue;
            }
        }
        throw new ShifuException(ShifuErrorCode.ERROR_GRID_SEARCH_FILE_CONFIG);
    }

    /**
     * Hard Coded!
     * 
     * @param key
     *            param name/key
     * @return
     *         the complete json path joined by '#', which aligns with {@link MetaFactory}
     */
    private String getItemKeyInMeta(String key) {
        return "train" + MetaFactory.ITEM_KEY_SEPERATOR + "params" + MetaFactory.ITEM_KEY_SEPERATOR + key;
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
