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
package ml.shifu.core.util;

import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import ml.shifu.core.container.*;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.Map.Entry;

/**
 * {@link CommonUtils} is used to for almost all kinds of utility function in this framework.
 */
public final class CommonUtils {

    private static final Logger log = LoggerFactory.getLogger(CommonUtils.class);

    /**
     * Avoid using new for our utility class.
     */
    private CommonUtils() {
    }

    /**
     * Return final selected column collection.
     */
    public static Collection<ColumnConfig> getFinalSelectColumnConfigList(Collection<ColumnConfig> columnConfigList) {
        return Collections2.filter(columnConfigList, new com.google.common.base.Predicate<ColumnConfig>() {
            @Override
            public boolean apply(ColumnConfig input) {
                return input.isFinalSelect();
            }
        });
    }


    /**
     * Get relative column name from pig header. For example, one column is a::b, return b. If b, return b.
     *
     * @throws NullPointerException if parameter raw is null.
     */
    public static String getRelativePigHeaderColumnName(String raw) {
        int position = raw.lastIndexOf(Constants.PIG_COLUMN_SEPARATOR);
        return position >= 0 ? raw.substring(position + Constants.PIG_COLUMN_SEPARATOR.length()) : raw;
    }

    /**
     * Given a column value, return bin list index. Return 0 for Category because of index 0 is started from
     * NEGATIVE_INFINITY.
     *
     * @throws IllegalArgumentException if input is null or empty.
     * @throws NumberFormatException    if columnVal does not contain a parsable number.
     */
    public static int getBinNum(ColumnConfig columnConfig, Object columnVal) {
        if (StringUtils.isEmpty(columnVal.toString()) || columnConfig == null) {
            throw new IllegalArgumentException(
                    String.format(
                            "columnVal should not be null or empty, columnConfig should not be null, columnVal:%s, columnConfig:%s",
                            columnVal, columnConfig)
            );
        }

        if (columnConfig.isCategorical()) {
            List<String> binCategories = columnConfig.getBinCategory();
            for (int i = 0; i < binCategories.size(); i++) {
                if (binCategories.get(i).equals(columnVal)) {
                    return i;
                }
            }
            return 0;
        } else {
            return getNumericBinNum(columnConfig.getBinBoundary(), Double.valueOf(columnVal.toString()));
        }
    }

    /**
     * Return the real bin number for one value. As the first bin value is NEGATIVE_INFINITY, invalid index is 0, not
     * -1.
     *
     * @param binBoundary bin boundary list which should be sorted.
     * @throws IllegalArgumentException if binBoundary is null or empty.
     */
    private static int getNumericBinNum(List<Double> binBoundary, double value) {
        if (CollectionUtils.isEmpty(binBoundary)) {
            throw new IllegalArgumentException("binBoundary should not be null or empty.");
        }

        int n = binBoundary.size() - 1;
        while (n > 0 && value < binBoundary.get(n)) {
            n--;
        }
        return n;
    }

    /**
     * Common split function to ignore special character like '|'. It's better to return a list while many calls in our
     * framework using string[].
     *
     * @throws IllegalArgumentException {@code raw} and {@code delimiter} is null or empty.
     */
    public static String[] split(String raw, String delimiter) {
        List<String> split = splitAndReturnList(raw, delimiter);
        return split.toArray(new String[split.size()]);
    }

    /**
     * Common split function to ignore special character like '|'.
     *
     * @throws IllegalArgumentException {@code raw} and {@code delimiter} is null or empty.
     */
    public static List<String> splitAndReturnList(String raw, String delimiter) {
        if (StringUtils.isEmpty(raw) || StringUtils.isEmpty(delimiter)) {
            throw new IllegalArgumentException(String.format(
                    "raw and delimiter should not be null or empty, raw:%s, delimiter:%s", raw, delimiter));
        }
        List<String> headerList = new ArrayList<String>();
        for (String str : Splitter.on(delimiter).split(raw)) {
            headerList.add(str);
        }

        return headerList;

    }

    /**
     * Get target column.
     *
     * @throws IllegalArgumentException if columnConfigList is null or empty.
     * @throws IllegalStateException    if no target column can be found.
     */
    public static Integer getTargetColumnNum(List<ColumnConfig> columnConfigList) {
        if (CollectionUtils.isEmpty(columnConfigList)) {
            throw new IllegalArgumentException("columnConfigList should not be null or empty.");
        }
        // I need cast operation because of common-collections doesn't support generic.
        ColumnConfig cc = (ColumnConfig) CollectionUtils.find(columnConfigList, new Predicate() {
            @Override
            public boolean evaluate(Object object) {
                return ((ColumnConfig) object).isTarget();
            }
        });
        if (cc == null) {
            throw new IllegalStateException("No target column can be found, please check your column configurations");
        }
        return cc.getColumnNum();
    }


    /**
     * Return one HashMap Object contains keys in the first parameter, values in the second parameter. Before calling
     * this method, you should be aware that headers should be unique.
     *
     * @throws IllegalArgumentException if lengths of two arrays are not the same.
     * @throws NullPointerException     if header or data is null.
     */
    public static Map<String, String> getRawDataMap(String[] header, String[] data) {
        if (header.length != data.length) {
            throw new IllegalArgumentException(String.format("Header/Data mismatch: Header length %s, Data length %s",
                    header.length, data.length));
        }

        Map<String, String> rawDataMap = new HashMap<String, String>(header.length);
        for (int i = 0; i < header.length; i++) {
            rawDataMap.put(header[i], data[i]);
        }
        return rawDataMap;
    }


    /**
     * Change list str to List object with double type.
     *
     * @throws IllegalArgumentException if str is not a valid list str: [1,2].
     */
    public static List<Double> stringToDoubleList(String str) {
        List<String> list = checkAndReturnSplitCollections(str);

        return Lists.transform(list, new Function<String, Double>() {
            @Override
            public Double apply(String input) {
                return Double.valueOf(input.trim());
            }
        });
    }

    private static List<String> checkAndReturnSplitCollections(String str) {
        checkListStr(str);
        return Arrays.asList(str.trim().substring(1, str.length() - 1).split(Constants.COMMA));
    }

    private static void checkListStr(String str) {
        if (StringUtils.isEmpty(str)) {
            throw new IllegalArgumentException("str should not be null or empty");
        }
        if (!str.startsWith("[") || !str.endsWith("]")) {
            throw new IllegalArgumentException("Invalid list string format, should be like '[1,2,3]'");
        }
    }

    /**
     * Change list str to List object with integer type.
     *
     * @throws IllegalArgumentException if str is not a valid list str.
     */
    public static List<Integer> stringToIntegerList(String str) {
        List<String> list = checkAndReturnSplitCollections(str);
        return Lists.transform(list, new Function<String, Integer>() {
            @Override
            public Integer apply(String input) {
                return Integer.valueOf(input.trim());
            }
        });
    }

    /**
     * Change list str to List object with string type.
     *
     * @throws IllegalArgumentException if str is not a valid list str.
     */
    public static List<String> stringToStringList(String str) {
        List<String> list = checkAndReturnSplitCollections(str);
        return Lists.transform(list, new Function<String, String>() {
            @Override
            public String apply(String input) {
                return input.trim();
            }
        });
    }

    /**
     * Return map entries sorted by value.
     */
    public static <K, V extends Comparable<V>> List<Map.Entry<K, V>> getEntriesSortedByValues(Map<K, V> map) {
        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {
            @Override
            public int compare(Entry<K, V> o1, Entry<K, V> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });

        return entries;
    }


    /**
     * Expanding score by expandingFactor
     */
    public static long getExpandingScore(double d, int expandingFactor) {
        return Math.round(d * expandingFactor);
    }

    /**
     * Return column name string with 'derived_' started
     *
     * @throws NullPointerException if modelConfig is null or columnConfigList is null.
     */
    public static List<String> getDerivedColumnNames(List<ColumnConfig> columnConfigList) {
        List<String> derivedColumnNames = new ArrayList<String>();

        for (ColumnConfig config : columnConfigList) {
            if (config.getColumnName().startsWith(Constants.DERIVED)) {
                derivedColumnNames.add(config.getColumnName());
            }
        }
        return derivedColumnNames;
    }

    /**
     * Get the file separator regex
     *
     * @return "/" - if the OS is Linux
     * "\\\\" - if the OS is Windows
     */
    public static String getPathSeparatorRegex() {
        if (File.separator.equals(Constants.SLASH)) {
            return File.separator;
        } else {
            return Constants.BACK_SLASH + File.separator;
        }
    }


    /**
     * To check whether there is targetColumn in columns or not
     *
     * @return true - if the columns contains targetColumn, or false
     */
    public static boolean isColumnExists(String[] columns, String targetColumn) {
        if (ArrayUtils.isEmpty(columns) || StringUtils.isBlank(targetColumn)) {
            return false;
        }

        for (String column : columns) {
            if (column != null && column.equalsIgnoreCase(targetColumn)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Returns the element if it is in both collections.
     * - return null if any collection is null or empty
     * - return null if no element exists in both collections
     *
     * @param leftCol  - left collection
     * @param rightCol - right collection
     * @return First element that are found in both collections
     * null if no elements in both collection or any collection is null or empty
     */
    public static <T> T containsAny(Collection<T> leftCol, Collection<T> rightCol) {
        if (CollectionUtils.isEmpty(leftCol) || CollectionUtils.isEmpty(rightCol)) {
            return null;
        }

        for (T element : leftCol) {
            if (rightCol.contains(element)) {
                return element;
            }
        }

        return null;
    }

    /**
     * Escape the delimiter for Pig.... Since the Pig doesn't support invisible character
     *
     * @param delimiter - the original delimiter
     * @return the delimiter after escape
     */
    public static String escapePigString(String delimiter) {
        StringBuilder buf = new StringBuilder();

        for (int i = 0; i < delimiter.length(); i++) {
            char c = delimiter.charAt(i);
            switch (c) {
                case '\t':
                    buf.append("\\\\t");
                    break;
                default:
                    buf.append(c);
                    break;
            }
        }

        return buf.toString();
    }


    /**
     * Generate seat info for selected column in @columnConfigList
     *
     * @param columnConfigList
     * @return
     */
    public static Map<String, Integer> generateColumnSeatMap(List<ColumnConfig> columnConfigList) {
        List<ColumnConfig> selectedColumnList = new ArrayList<ColumnConfig>();
        for (ColumnConfig columnConfig : columnConfigList) {
            if (columnConfig.isFinalSelect()) {
                selectedColumnList.add(columnConfig);
            }
        }
        Collections.sort(selectedColumnList, new Comparator<ColumnConfig>() {
            @Override
            public int compare(ColumnConfig from, ColumnConfig to) {
                return from.getColumnName().compareTo(to.getColumnName());
            }

        });

        Map<String, Integer> columnSeatMap = new HashMap<String, Integer>();
        for (int i = 0; i < selectedColumnList.size(); i++) {
            columnSeatMap.put(selectedColumnList.get(i).getColumnName(), i);
        }

        return columnSeatMap;
    }

    /**
     * Find the @ColumnConfig according the column name
     *
     * @param columnConfigList
     * @param columnName
     * @return
     */
    public static ColumnConfig findColumnConfigByName(List<ColumnConfig> columnConfigList, String columnName) {
        for (ColumnConfig columnConfig : columnConfigList) {
            if (columnConfig.getColumnName().equalsIgnoreCase(columnName)) {
                return columnConfig;
            }
        }
        return null;
    }

    /**
     * Convert data into <key, value> map. The @inputData is String of a record, which is delimited by @delimiter
     * If fields in @inputData is not equal @header size, return null
     *
     * @param inputData - String of a record
     * @param delimiter - the delimiter of the input data
     * @param header    - the column names for all the input data
     * @return <key, value> map for the record
     */
    public static Map<String, String> convertDataIntoMap(String inputData, String delimiter, String[] header) {
        String[] input = CommonUtils.split(inputData, delimiter);
        if (input == null || input.length == 0 || input.length != header.length) {
            log.error("the wrong input data, {}", inputData);
            return null;
        }

        Map<String, String> rawDataMap = new HashMap<String, String>(input.length);
        for (int i = 0; i < header.length; i++) {
            if (input[i] == null) {
                rawDataMap.put(header[i], "");
            } else {
                rawDataMap.put(header[i], input[i]);
            }
        }

        return rawDataMap;
    }


    public static Class getClass(String name) {
        try {
            return Class.forName(name);

        } catch (Exception e) {
            throw new RuntimeException("No such implementation class: " + name);
        }
    }

    public static boolean isValidNumber(Object raw) {

        Double value;
        try {
            value = Double.parseDouble(raw.toString());
        } catch (Exception e) {
            return false;
        }

        return !(Double.isNaN(value) || Double.isInfinite(value));

    }

    public static List<NumericalValueObject> convertListRaw2Numerical(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {
        List<NumericalValueObject> nvoList = new ArrayList<NumericalValueObject>();

        for (RawValueObject rvo : rvoList) {
            NumericalValueObject nvo = new NumericalValueObject();

            // Set Value
            if (!CommonUtils.isValidNumber(rvo.getValue().toString())) {
                continue;
            }
            nvo.setValue(Double.valueOf(rvo.getValue().toString()));

            // Set Tag
            if (posTags.contains(rvo.getTag())) {
                nvo.setIsPositive(true);
            } else if (negTags.contains(rvo.getTag())) {
                nvo.setIsPositive(false);
            } else {
                // ignore
                continue;
            }

            // Set Weight
            nvo.setWeight(rvo.getWeight());
            nvoList.add(nvo);
        }

        return nvoList;
    }

    public static List<CategoricalValueObject> convertListRaw2Categorical(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {

        List<CategoricalValueObject> cvoList = new ArrayList<CategoricalValueObject>();

        for (RawValueObject rvo : rvoList) {
            CategoricalValueObject cvo = new CategoricalValueObject();

            // Set Value
            if (rvo.getValue() == null) {
                continue;
            }
            cvo.setValue(rvo.getValue().toString());

            // Set Tag
            if (posTags.contains(rvo.getTag())) {
                cvo.setIsPositive(true);
            } else if (negTags.contains(rvo.getTag())) {
                cvo.setIsPositive(false);
            } else {
                // ignore
                continue;
            }

            // Set Weight
            cvo.setWeight(rvo.getWeight());
            cvoList.add(cvo);
        }

        return cvoList;
    }

    public static List<NumericalValueObject> convertListCategorical2Numerical(List<CategoricalValueObject> cvoList, ColumnBinningResult columnBinningResult) {

        List<NumericalValueObject> nvoList = new ArrayList<NumericalValueObject>();

        for (CategoricalValueObject cvo : cvoList) {

            NumericalValueObject nvo = new NumericalValueObject();

            int index = columnBinningResult.getBinCategory().indexOf(cvo.getValue());

            nvo.setValue(columnBinningResult.getBinPosRate().get(index));
            nvo.setIsPositive(cvo.getIsPositive());
            nvo.setWeight(cvo.getWeight());

            nvoList.add(nvo);
        }

        return nvoList;

    }
}