/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.container.meta;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.math.DoubleMath;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.container.obj.ModelSourceDataConf;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.container.obj.ModelVarSelectConf;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.util.Constants;

/**
 * MetaFactory class
 * MetaFactory hosts all the meta for ModelConfig.
 * It provides the capability to validate ModelConfig object and sub-fields.
 * It can also provide deep copy of all meta
 */
public class MetaFactory {

    private final static Logger LOG = LoggerFactory.getLogger(MetaFactory.class);

    // flag to indicate validate result
    public static final String VALIDATE_OK = "OK";

    // key related variables
    public static final String ITEM_KEY_SEPERATOR = "#";
    public static final String DUMMY = "dummy";

    // the tags for group
    private static final String BASIC_TAG = "basic";
    private static final String DATASET_TAG = "dataSet";
    private static final String STATS_TAG = "stats";
    private static final String VARSELECT_TAG = "varSelect";
    private static final String NORMALIZE_TAG = "normalize";
    private static final String TRAIN_TAG = "train";
    private static final String EVALS_TAG = "evals";
    private static final String TRAIN_PARAM_TAG = "params";

    // default MetaConfig input file
    public static final String MODEL_META_STORE_FILE = "store/ModelConfigMeta.json";

    // warehouse for meta item
    private static Map<String, MetaItem> itemsWareHouse;

    /**
     * Load the MetalConfig into Memory, and organize them in flatten format so that user
     * can use 'key' to query MetaItem.
     * 
     * Please note, the key is composed by 'group'#'name'
     * 
     */
    static {
        ObjectMapper jsonMapper = new ObjectMapper();
        MetaGroup[] groups = null;
        try {
            groups = jsonMapper.readValue(MetaFactory.class.getClassLoader().getResource(MODEL_META_STORE_FILE),
                    MetaGroup[].class);
        } catch (Exception e) {
            throw new RuntimeException("Fail to read model meta from " + MODEL_META_STORE_FILE, e);
        }

        itemsWareHouse = new HashMap<String, MetaItem>();
        if(groups != null && groups.length > 0) {
            for(MetaGroup metaGroup: groups) {
                for(MetaItem metaItem: metaGroup.getMetaList()) {
                    String key = metaGroup.getGroup() + ITEM_KEY_SEPERATOR + metaItem.getName();
                    addMetaItem(key, metaItem);
                }
            }
        }
    }

    /**
     * Get a copy of itemsWarehouse. Deep copy, so that user couldn't change the content of constrain
     * 
     * @return a stand-alone copy of @itemsWarehouse
     */
    public static Map<String, MetaItem> getModelConfigMeta() {
        Map<String, MetaItem> retMap = new HashMap<String, MetaItem>();

        Iterator<Entry<String, MetaItem>> iterator = itemsWareHouse.entrySet().iterator();
        while(iterator.hasNext()) {
            Entry<String, MetaItem> entry = iterator.next();
            retMap.put(entry.getKey(), entry.getValue().clone());

        }

        return retMap;
    }

    /**
     * Validate the ModelConfig object, to make sure each item follow the constrain
     * 
     * @param modelConfig
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelConfig modelConfig) throws Exception {
        ValidateResult result = new ValidateResult(true);

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                modelConfig.getTrain().getGridConfigFileContent());

        Class<?> cls = modelConfig.getClass();
        Field[] fields = cls.getDeclaredFields();

        for(Field field: fields) {
            // skip log instance
            if(field.getName().equalsIgnoreCase("log") || field.getName().equalsIgnoreCase("logger")
                    || isJsonIgnoreField(field)) {
                continue;
            }
            if(!field.isSynthetic()) {
                Method method = cls.getMethod("get" + getMethodName(field.getName()));
                Object value = method.invoke(modelConfig);

                if(value instanceof List) {
                    List<?> objList = (List<?>) value;
                    for(Object obj: objList) {
                        encapsulateResult(result, iterateCheck(gs.hasHyperParam(), field.getName(), obj));
                    }
                } else {
                    encapsulateResult(result, iterateCheck(gs.hasHyperParam(), field.getName(), value));
                }
            }
        }

        return result;
    }

    /**
     * Validate the ModelBasicConf object, to make sure each item follow the constrain
     * 
     * @param basic
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelBasicConf basic) throws Exception {
        return iterateCheck(false, BASIC_TAG, basic);
    }

    /**
     * Validate the ModelSourceDataConf object, to make sure each item follow the constrain
     * 
     * @param sourceData
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelSourceDataConf sourceData) throws Exception {
        return iterateCheck(false, DATASET_TAG, sourceData);
    }

    /**
     * Validate the ModelStatsConf object, to make sure each item follow the constrain
     * 
     * @param stats
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelStatsConf stats) throws Exception {
        return iterateCheck(false, STATS_TAG, stats);
    }

    /**
     * Validate the ModelVarSelectConf object, to make sure each item follow the constrain
     * 
     * @param varselect
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelVarSelectConf varselect) throws Exception {
        return iterateCheck(false, VARSELECT_TAG, varselect);
    }

    /**
     * Validate the ModelNormalizeConf object, to make sure each item follow the constrain
     * 
     * @param normalizer
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelNormalizeConf normalizer) throws Exception {
        return iterateCheck(false, NORMALIZE_TAG, normalizer);
    }

    /**
     * Validate the ModelTrainConf object, to make sure each item follow the constrain
     * 
     * @param train
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(ModelTrainConf train) throws Exception {
        return iterateCheck(false, TRAIN_TAG, train);
    }

    /**
     * Validate the List(EvalConfig) object, to make sure each item follow the constrain
     * 
     * @param evalList
     *            - object list to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(List<EvalConfig> evalList) throws Exception {
        ValidateResult result = new ValidateResult(true);

        for(EvalConfig evalConfig: evalList) {
            encapsulateResult(result, validate(evalConfig));
        }

        return result;
    }

    /**
     * Validate the EvalConfig, to make sure each item follow the constrain
     * 
     * @param eval
     *            - object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult validate(EvalConfig eval) throws Exception {
        return iterateCheck(false, EVALS_TAG, eval);
    }

    /**
     * Iterate each property of Object, get the value and validate
     * 
     * @param isGridSearch
     *            - if grid search, ignore validation in train#params as they are set all as list
     * @param ptag
     *            - the prefix of key to search @MetaItem
     * @param obj
     *            - the object to validate
     * @return ValidateResult
     *         If all items are OK, the ValidateResult.status will be true;
     *         Or the ValidateResult.status will be false, ValidateResult.causes will contain the reasons
     * @throws Exception
     *             any exception in validaiton
     */
    public static ValidateResult iterateCheck(boolean isGridSearch, String ptag, Object obj) throws Exception {
        ValidateResult result = new ValidateResult(true);
        if(obj == null) {
            return result;
        }

        Class<?> cls = obj.getClass();
        Field[] fields = cls.getDeclaredFields();

        Class<?> parentCls = cls.getSuperclass();
        if(!parentCls.equals(Object.class)) {
            Field[] pfs = parentCls.getDeclaredFields();
            fields = (Field[]) ArrayUtils.addAll(fields, pfs);
        }

        for(Field field: fields) {
            if(!field.isSynthetic() && !Modifier.isStatic(field.getModifiers()) && !isJsonIgnoreField(field)) {
                Method method;
                try {
                    method = cls.getMethod("get" + getMethodName(field.getName()));
                    Object value = method.invoke(obj);

                    encapsulateResult(result,
                            validate(isGridSearch, ptag + ITEM_KEY_SEPERATOR + field.getName(), value));
                } catch (NoSuchMethodException e) {
                    LOG.error(
                            "No method get" + getMethodName(field.getName()) + " in class " + obj.getClass().getName(),
                            e);
                    throw e;
                }
            }
        }

        return result;
    }

    private static boolean isJsonIgnoreField(Field field) {
        Annotation[] annotations = field.getAnnotations();
        for(Annotation annotation: annotations) {
            if(annotation.annotationType().getName().equals(JsonIgnore.class.getName())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Filter out train params because param value may be set to list to enable grid search.
     * Validation for train params is done at {@link GridSearch}.
     * 
     * @param itemKey
     *            - the key to locate MetaItem
     * @return
     */
    private static boolean filterOut(String itemKey) {
        return itemKey.matches(TRAIN_TAG + ITEM_KEY_SEPERATOR + TRAIN_PARAM_TAG + ITEM_KEY_SEPERATOR + ".+");
    }

    /**
     * Validate the input value. Find the @MetaItem from warehouse, and do the validation
     * 
     * @param isGridSearch
     *            - if grid search, ignore validation in train#params as they are set all as list
     * @param itemKey
     *            - the key to locate MetaItem
     * @param itemValue
     *            - the value to validate
     * @return if validate OK, return "OK"
     *         or return the cause - String
     * @throws Exception
     *             any exception in validaiton
     */
    public static String validate(boolean isGridSearch, String itemKey, Object itemValue) throws Exception {
        MetaItem itemMeta = itemsWareHouse.get(itemKey);

        if(isGridSearch && filterOut(itemKey)) {
            return VALIDATE_OK;
        }

        if(itemMeta == null) {
            return itemKey + " - not found meta info.";
        }

        if(itemValue == null && itemMeta.getNotNull()) {
            return itemKey + " - the value couldn't be null.";
        }

        if(itemMeta.getType().equals("text")) {
            String value = ((itemValue == null) ? null : itemValue.toString());

            if(itemMeta.getMaxLength() != null && value != null && value.length() > itemMeta.getMaxLength()) {
                return itemKey + " - the length of value exceeds the max length : " + itemMeta.getMaxLength();
            }

            if(itemMeta.getMinLength() != null && (value == null || value.length() < itemMeta.getMinLength())) {
                if(value == null) {
                    return itemKey + " - then shouldn't be null";
                } else {
                    return itemKey + " - the length of value less than min length : " + itemMeta.getMinLength();
                }
            }

            if(CollectionUtils.isNotEmpty(itemMeta.getOptions())) {
                boolean isOptionValue = false;
                for(ValueOption itemOption: itemMeta.getOptions()) {
                    String optValue = (String) itemOption.getValue();
                    if(optValue.equalsIgnoreCase(value)) {
                        isOptionValue = true;
                        break;
                    }
                }

                if(!isOptionValue) {
                    return itemKey + " - the value couldn't be found in the option value list - "
                            + convertOptionIntoString(itemMeta.getOptions());
                }
            }
        } else if(itemMeta.getType().equals("integer") || itemMeta.getType().equals("int")) {
            if(itemValue == null) {
                if(CollectionUtils.isNotEmpty(itemMeta.getOptions())) {
                    return itemKey + " - the value couldn't be null.";
                }
            } else {
                Integer value = null;
                try {
                    value = Integer.valueOf(itemValue.toString());
                } catch (NumberFormatException e) {
                    return itemKey + " - the value is not integer format.";
                }

                if(value != null && CollectionUtils.isNotEmpty(itemMeta.getOptions())) {
                    boolean isOptionValue = false;
                    for(ValueOption itemOption: itemMeta.getOptions()) {
                        Integer optValue = Integer.valueOf(itemOption.getValue().toString());
                        if(value.equals(optValue)) {
                            isOptionValue = true;
                            break;
                        }
                    }

                    if(!isOptionValue) {
                        return itemKey + " - the value couldn't be found in the option value list - "
                                + convertOptionIntoString(itemMeta.getOptions());
                    }
                }
            }
        } else if(itemMeta.getType().equals("number") || itemMeta.getType().equals("float")) {
            if(itemValue == null) {
                if(CollectionUtils.isNotEmpty(itemMeta.getOptions())) {
                    return itemKey + " - the value couldn't be null.";
                }
            } else {
                Double value = null;
                try {
                    value = Double.valueOf(itemValue.toString());
                } catch (NumberFormatException e) {
                    return itemKey + " - the value is not number format.";
                }

                if(value != null && CollectionUtils.isNotEmpty(itemMeta.getOptions())) {
                    boolean isOptionValue = false;
                    for(ValueOption itemOption: itemMeta.getOptions()) {
                        Double optValue = Double.valueOf(itemOption.getValue().toString());
                        if(DoubleMath.fuzzyEquals(value, optValue, Constants.TOLERANCE)) {
                            isOptionValue = true;
                            break;
                        }
                    }

                    if(!isOptionValue) {
                        return itemKey + " - the value couldn't be found in the option value list - "
                                + convertOptionIntoString(itemMeta.getOptions());
                    }
                }
            }
        } else if(itemMeta.getType().equals("boolean")) {
            if(itemValue == null) {
                return itemKey + " - the value couldn't be null. Only true/false are perimited.";
            }

            if(!itemValue.toString().equalsIgnoreCase("true") && !itemValue.toString().equalsIgnoreCase("false")) {
                return itemKey + " - the value is illegal.  Only true/false are perimited.";
            }
        } else if(itemMeta.getType().equals("list")) {
            if(itemValue != null && itemMeta.getElement() != null) {
                @SuppressWarnings("unchecked")
                List<Object> valueList = (List<Object>) itemValue;

                for(Object obj: valueList) {
                    if(itemMeta.getElementType().equals("object")) {
                        ValidateResult result = iterateCheck(isGridSearch, itemKey, obj);
                        if(!result.getStatus()) {
                            return result.getCauses().get(0);
                        }
                    } else {
                        String validateStr = validate(isGridSearch, itemKey + ITEM_KEY_SEPERATOR + DUMMY, obj);
                        if(!validateStr.equals(VALIDATE_OK)) {
                            return validateStr;
                        }
                    }
                }
            }
        } else if(itemMeta.getType().equals("map")) {
            if(itemValue != null && itemMeta.getElement() != null) {
                @SuppressWarnings("unchecked")
                Map<String, Object> valueMap = (Map<String, Object>) itemValue;

                Iterator<Entry<String, Object>> iterator = valueMap.entrySet().iterator();
                while(iterator.hasNext()) {
                    Entry<String, Object> entry = iterator.next();
                    String key = entry.getKey();
                    Object value = entry.getValue();

                    String validateStr = validate(isGridSearch, itemKey + ITEM_KEY_SEPERATOR + key, value);
                    if(!validateStr.equals(VALIDATE_OK)) {
                        return validateStr;
                    }
                }
            }
        }

        return VALIDATE_OK;
    }

    /**
     * Add the MetaItem into warehouse.
     * If the type of MetaItem is list, try to add the child elements
     * 
     * @param key
     *            - the key to store MetaItem
     * @param metaItem
     *            - object to store
     */
    private static void addMetaItem(String key, MetaItem metaItem) {
        itemsWareHouse.put(key, metaItem);

        if(StringUtils.equals(metaItem.getType(), "list")) {
            if(StringUtils.equals(metaItem.getElementType(), "object")) {
                if(CollectionUtils.isNotEmpty(metaItem.getElement())) {
                    for(MetaItem sub: metaItem.getElement()) {
                        addMetaItem(key + ITEM_KEY_SEPERATOR + sub.getName(), sub);
                    }
                }
            } else {
                if(CollectionUtils.isNotEmpty(metaItem.getElement())) {
                    MetaItem sub = metaItem.getElement().get(0);
                    addMetaItem(key + ITEM_KEY_SEPERATOR + DUMMY, sub);
                }
            }
        } else if(StringUtils.equals(metaItem.getType(), "map")) {
            if(CollectionUtils.isNotEmpty(metaItem.getElement())) {
                for(MetaItem sub: metaItem.getElement()) {
                    addMetaItem(key + ITEM_KEY_SEPERATOR + sub.getName(), sub);
                }
            }
        } else if(StringUtils.equals(metaItem.getType(), "object")) {
            if(CollectionUtils.isNotEmpty(metaItem.getElement())) {
                for(MetaItem sub: metaItem.getElement()) {
                    addMetaItem(key + ITEM_KEY_SEPERATOR + sub.getName(), sub);
                }
            }
        }
    }

    /**
     * Convert the value of ValueOption list into String
     * For example, if the mode options are ["local", "hdfs"], the output will be local/hdfs
     * 
     * @param options
     *            - ValueOption list
     * @return - String of ValueOption list, separated by '/'
     */
    private static String convertOptionIntoString(List<ValueOption> options) {
        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < options.size(); i++) {
            if(i > 0) {
                builder.append("/");
            }
            builder.append(options.get(i).getValue().toString());
        }
        return builder.toString();
    }

    /**
     * Encapsulate the validate result string into @ValidateResult.
     * If the validateStr is not "OK", set the ValidateResult.status to false, and add @validateStr into
     * ValidateResult.causes
     * 
     * @param result
     *            - result set
     * @param validateStr
     *            - validate result string
     */
    private static void encapsulateResult(ValidateResult result, String validateStr) {
        if(result != null) {
            if(!VALIDATE_OK.equals(validateStr)) {
                result.setStatus(false);
                result.getCauses().add(validateStr);
            }
        }
    }

    /**
     * Encapsulate validate result into total result.
     * The status of total result will be false, if there is one false.
     * The total result will contain all causes
     * 
     * @param totalResult
     *            the total result
     * @param result
     *            the current result
     */
    private static void encapsulateResult(ValidateResult totalResult, ValidateResult result) {
        totalResult.setStatus(totalResult.getStatus() && result.getStatus());
        totalResult.getCauses().addAll(result.getCauses());
    }

    /**
     * Get the method-style name of the property. (UPPER the first character:))
     * 
     * @param fieldName
     *            the field name
     * @return first character Upper style
     */
    private static String getMethodName(String fieldName) {
        return StringUtils.capitalize(fieldName);
    }
}
