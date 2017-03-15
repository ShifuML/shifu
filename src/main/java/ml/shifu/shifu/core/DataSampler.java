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
package ml.shifu.shifu.core;

import java.util.List;
import java.util.Random;
import java.util.Set;

import ml.shifu.shifu.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * DataSampler class
 * - Output: column 0 is tag, following final select vars
 */
public class DataSampler {
    private static Logger log = LoggerFactory.getLogger(DataSampler.class);
    private static Random rd = new Random(System.currentTimeMillis());

    /**
     * check whether the data should be filtered out or not
     * the data will be filtered out if
     * - the target value is invalid
     * - or target tag is not in positive tag list or negative tag list
     * - or not be sampled
     * 
     * @param targetColumnNum
     *            the target column
     * @param posTags
     *            posTags
     * @param negTags
     *            negTags
     * @param data
     *            data
     * @param sampleRate
     *            sampleRate
     * @param sampleNegOnly
     *            sampleNegOnly
     * @return null - if the data should be filtered out
     *         data itself - if the data should not be filtered out
     */
    public static List<Object> filter(Integer targetColumnNum, List<String> posTags, List<String> negTags,
            List<Object> data, Double sampleRate, Boolean sampleNegOnly) {
        String tag = CommonUtils.trimTag(data.get(targetColumnNum).toString());

        if(isNotSampled(posTags, negTags, sampleRate, sampleNegOnly, tag)) {
            return null;
        }

        return data;
    }

    /**
     * check whether the fields should be filtered out or not
     * the data will be filtered out if
     * - the target value is invalid
     * - or target tag is not in positive tag list or negative tag list
     * - or not be sampled
     * 
     * @param targetColumnNum
     *            the target column
     * @param posTags
     *            posTags
     * @param negTags
     *            negTags
     * @param fields
     *            fields
     * @param sampleRate
     *            sampleRate
     * @param sampleNegOnly
     *            sampleNegOnly
     * @return true - if the data should be filtered out
     *         false - if the data should not be filtered out
     */
    public static boolean filter(int targetColumnNum, List<String> posTags, List<String> negTags, String[] fields,
            double sampleRate, boolean sampleNegOnly) {
        String tag = CommonUtils.trimTag(fields[targetColumnNum]);
        return isNotSampled(posTags, negTags, sampleRate, sampleNegOnly, tag);
    }

    /**
     * To decide whether the data should be filtered out or not. Both unselected data or invalid tag will be
     * filtered out.
     * 
     * @param posTags
     *            posTags
     * @param negTags
     *            negTags
     * @param sampleRate
     *            sampleRate
     * @param sampleNegOnly
     *            sampleNegOnly
     * @param tag
     *            tag
     * @return true - if the data should be filtered out
     *         false - if the data should not be filtered out
     */
    public static boolean isNotSampled(List<String> posTags, List<String> negTags, double sampleRate,
            boolean sampleNegOnly, String tag) {
        if(tag == null) {
            log.error("Tag is null.");
            return true;
        }

        if(!(posTags.contains(tag) || negTags.contains(tag))) {
            log.error("Invalid target column value - " + tag);
            return true;
        }

        if(sampleNegOnly) {
            return (negTags.contains(tag) && rd.nextDouble() > sampleRate);
        } else {
            return (rd.nextDouble() > sampleRate);
        }
    }

    public static boolean isNotSampled(boolean isBinary, Set<String> tags, Set<String> posTags, Set<String> negTags,
            double sampleRate, boolean sampleNegOnly, String tag) {
        if(tag == null) {
            log.error("Tag is null.");
            return true;
        }

        if(!isBinary && !tags.contains(tag)) {
            log.error("Invalid target column value - " + tag);
            return true;
        }
        if(isBinary && !(posTags.contains(tag) || negTags.contains(tag))) {
            log.error("Invalid target column value - " + tag);
            return true;
        }

        if(isBinary && sampleNegOnly) {
            return (negTags.contains(tag) && rd.nextDouble() > sampleRate);
        } else {
            return (rd.nextDouble() > sampleRate);
        }
    }
}
