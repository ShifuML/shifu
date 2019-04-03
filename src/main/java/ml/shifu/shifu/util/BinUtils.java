/*
 * Copyright [2012-2018] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import com.google.common.base.Splitter;
import ml.shifu.shifu.container.obj.ColumnConfig;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * {@link BinUtils} is used to for almost all kinds of utility function in this framework.
 */
public final class BinUtils {

    /**
     * Avoid using new for our utility class.
     */
    private BinUtils() {
    }

    /**
     * Given a column value, return bin list index. Return 0 for Category because of index 0 is started from
     * NEGATIVE_INFINITY.
     * 
     * @param columnConfig
     *            column config
     * @param columnVal
     *            value of the column
     * @return bin index of than value
     * @throws IllegalArgumentException
     *             if input is null or empty.
     * 
     * @throws NumberFormatException
     *             if columnVal does not contain a parsable number.
     */
    public static int getBinNum(ColumnConfig columnConfig, Object columnVal) {
        if(columnConfig.isCategorical()) {
            if(columnVal == null) {
                return -1;
            } else {
                return getCategoicalBinIndex(columnConfig, columnVal.toString());
            }
        } else {
            return getNumericalBinIndex(columnConfig.getBinBoundary(), columnVal);
        }
    }

    /**
     * Get numerical bin index according to string column value.
     * 
     * @param binBoundaries
     *            the bin boundaries
     * @param columnVal
     *            the column value
     * @return bin index, -1 if invalid values
     */
    public static int getNumericalBinIndex(List<Double> binBoundaries, Object columnVal) {
        if(columnVal == null) {
            return -1;
        }

        double dval = 0.0;

        if(columnVal instanceof Double) {
            dval = (Double) columnVal;
        } else if(columnVal instanceof Integer) {
            dval = ((Integer) columnVal).doubleValue();
        } else {
            try {
                dval = Double.parseDouble(columnVal.toString());
            } catch (Exception e) {
                return -1;
            }
        }
        return getBinIndex(binBoundaries, dval);
    }

    /**
     * Get categorical bin index according to string column value.
     * 
     * @param columnConfig
     *            the column config
     * @param columnVal
     *            the column value
     * @return bin index, -1 if invalid values
     */
    public static int getCategoicalBinIndex(ColumnConfig columnConfig, String columnVal) {
        if(StringUtils.isBlank(columnVal)) {
            return -1;
        }

        if(columnConfig.getColumnBinning().getBinCateMap() != null) {
            Map<String, Integer> binCateMap = columnConfig.getColumnBinning().getBinCateMap();
            Integer intIndex = binCateMap.get(columnVal);
            if(intIndex == null || intIndex < 0) {
                intIndex = -1;
            }
            return intIndex;
        } else {
            List<String> binCategories = columnConfig.getColumnBinning().getBinCategory();
            for(int i = 0; i < binCategories.size(); i++) {
                if(isCategoricalBinValue(binCategories.get(i), columnVal)) {
                    return i;
                }
            }
        }
        return -1;
    }

    /**
     * Get categorical bin index according to string column value.
     * 
     * @param binCategories
     *            the bin categories
     * @param columnVal
     *            the column value
     * @return bin index, -1 if invalid values
     */
    public static int getCategoicalBinIndex(List<String> binCategories, String columnVal) {
        if(StringUtils.isBlank(columnVal) || CollectionUtils.isEmpty(binCategories)) {
            return -1;
        }
        for(int i = 0; i < binCategories.size(); i++) {
            if(isCategoricalBinValue(binCategories.get(i), columnVal)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Check some categorical value is in the categorical value group or not
     * 
     * @param binVal
     *            - categorical value group, the format is lik cn^us^uk^jp
     * @param cval
     *            - categorical value to look up
     * @return true if the categorical value exists in group, else false
     */
    public static boolean isCategoricalBinValue(String binVal, String cval) {
        // TODO cache flattenCatValGrp(binVal)??
        return binVal.equals(cval) ? true : flattenCatValGrp(binVal).contains(cval);
    }

    /**
     * flatten categorical value group into values list
     * 
     * @param categoricalValGrp
     *            - categorical val group, it some values like zn^us^ck^
     * @return value list of categorical val
     */
    private static List<String> flattenCatValGrp(String categoricalValGrp) {
        List<String> catVals = new ArrayList<String>();
        if(StringUtils.isNotBlank(categoricalValGrp)) {
            for(String cval: Splitter.on(Constants.CATEGORICAL_GROUP_VAL_DELIMITER).split(categoricalValGrp)) {
                catVals.add(cval);
            }
        }
        return catVals;
    }

    /**
     * Get bin index by binary search. The last bin in <code>binBoundary</code> is missing value bin.
     * 
     * @param binBoundary
     *            bin boundary list which should be sorted.
     * @param dVal
     *            value of column
     * @return bin index
     */
    public static int getBinIndex(List<Double> binBoundary, Double dVal) {
        assert binBoundary != null && binBoundary.size() > 0;
        assert dVal != null;
        int binSize = binBoundary.size();

        int low = 0;
        int high = binSize - 1;

        while(low <= high) {
            int mid = (low + high) >>> 1;
            Double midVal = binBoundary.get(mid);
            int cmp = midVal.compareTo(dVal);

            if(cmp < 0) {
                low = mid + 1;
            } else if(cmp > 0) {
                high = mid - 1;
            } else {
                return mid; // key found
            }
        }

        return low == 0 ? 0 : low - 1;
    }

    /**
     * Avoid parsing times, failed parsing is set to NaN
     * 
     * @param valStr
     *            param string
     * @return double after parsing
     */
    public static double parseNumber(Object valStr) {
        if(valStr == null) {
            return Double.NaN;
        }

        if(valStr instanceof Double) {
            return (Double) valStr;
        } else if(valStr instanceof Integer) {
            return ((Integer) valStr).doubleValue();
        } else {
            try {
                return Double.parseDouble((String) valStr);
            } catch (NumberFormatException e) {
                return Double.NaN;
            }
        }
    }

}