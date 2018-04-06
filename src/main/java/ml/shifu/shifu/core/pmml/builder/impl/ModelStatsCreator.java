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
package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.*;
import org.encog.ml.BasicML;

import java.util.*;

/**
 * Created by zhanhu on 3/29/16.
 */
public class ModelStatsCreator extends AbstractPmmlElementCreator<ModelStats> {

    private static final double EPS = 1e-10;

    public ModelStatsCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public ModelStatsCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    @Override
    public ModelStats build(BasicML basicML) {
        ModelStats modelStats = new ModelStats();

        if(basicML instanceof BasicFloatNetwork) {
            BasicFloatNetwork bfn = (BasicFloatNetwork) basicML;
            Set<Integer> featureSet = bfn.getFeatureSet();
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.isFinalSelect()
                        && (CollectionUtils.isEmpty(featureSet) || featureSet.contains(columnConfig.getColumnNum()))) {
                    UnivariateStats univariateStats = new UnivariateStats();
                    // here, no need to consider if column is in segment expansion as we need to address new stats
                    // variable
                    univariateStats.setField(FieldName.create(CommonUtils.getSimpleColumnName(columnConfig
                            .getColumnName())));

                    if(columnConfig.isCategorical()) {
                        DiscrStats discrStats = new DiscrStats();

                        Array countArray = createCountArray(columnConfig);
                        discrStats.withArrays(countArray);

                        if(!isConcise) {
                            List<Extension> extensions = createExtensions(columnConfig);
                            discrStats.withExtensions(extensions);
                        }

                        univariateStats.setDiscrStats(discrStats);
                    } else { // numerical column
                        univariateStats.setNumericInfo(createNumericInfo(columnConfig));

                        if(!isConcise) {
                            univariateStats.setContStats(createConStats(columnConfig));
                        }
                    }

                    modelStats.withUnivariateStats(univariateStats);
                }
            }
        } else {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.isFinalSelect()) {
                    UnivariateStats univariateStats = new UnivariateStats();
                    // here, no need to consider if column is in segment expansion as we need to address new stats
                    // variable
                    univariateStats.setField(FieldName.create(CommonUtils.getSimpleColumnName(columnConfig
                            .getColumnName())));

                    if(columnConfig.isCategorical()) {
                        DiscrStats discrStats = new DiscrStats();

                        Array countArray = createCountArray(columnConfig);
                        discrStats.withArrays(countArray);

                        if(!isConcise) {
                            List<Extension> extensions = createExtensions(columnConfig);
                            discrStats.withExtensions(extensions);
                        }

                        univariateStats.setDiscrStats(discrStats);
                    } else { // numerical column
                        univariateStats.setNumericInfo(createNumericInfo(columnConfig));

                        if(!isConcise) {
                            univariateStats.setContStats(createConStats(columnConfig));
                        }
                    }

                    modelStats.withUnivariateStats(univariateStats);
                }
            }
        }

        return modelStats;
    }

    /**
     * Create @Array for numerical variable
     * 
     * @param columnConfig
     *            - ColumnConfig for numerical variable
     * @return Array for numerical variable ( positive count + negative count )
     */
    private Array createCountArray(ColumnConfig columnConfig) {
        Array countAllArray = new Array();

        List<Integer> binCountAll = new ArrayList<Integer>(columnConfig.getBinCountPos().size());
        for(int i = 0; i < binCountAll.size(); i++) {
            binCountAll.add(columnConfig.getBinCountPos().get(i) + columnConfig.getBinCountNeg().get(i));
        }

        countAllArray.setType(Array.Type.INT);
        countAllArray.setN(binCountAll.size());
        countAllArray.setValue(StringUtils.join(binCountAll, ' '));

        return countAllArray;
    }

    /**
     * Create common extension list from ColumnConfig
     * 
     * @param columnConfig
     *            - ColumnConfig to create extension
     * @return extension list
     */
    private List<Extension> createExtensions(ColumnConfig columnConfig) {
        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", columnConfig.getBinCountPos().toString());
        extensionMap.put("BinCountNeg", columnConfig.getBinCountNeg().toString());
        extensionMap.put("BinWeightedCountPos", columnConfig.getBinWeightedPos().toString());
        extensionMap.put("BinWeightedCountNeg", columnConfig.getBinWeightedNeg().toString());
        extensionMap.put("BinPosRate", columnConfig.getBinPosRate().toString());

        return createExtensions(extensionMap);
    }

    /**
     * Create extension list from HashMap
     * 
     * @param extensionMap
     *            the <String,String> map to create extension list
     * @return extension list
     */
    private List<Extension> createExtensions(Map<String, String> extensionMap) {
        List<Extension> extensions = new ArrayList<Extension>();

        for(Map.Entry<String, String> entry: extensionMap.entrySet()) {
            String key = entry.getKey();
            Extension extension = new Extension();
            extension.setName(key);
            extension.setValue(entry.getValue());
            extensions.add(extension);
        }

        return extensions;
    }

    /**
     * Create @NumericInfo for numerical variable
     * 
     * @param columnConfig
     *            - ColumnConfig for numerical variable
     * @return NumericInfo for variable
     */
    private NumericInfo createNumericInfo(ColumnConfig columnConfig) {
        NumericInfo numericInfo = new NumericInfo();

        numericInfo.setMaximum(columnConfig.getColumnStats().getMax());
        numericInfo.setMinimum(columnConfig.getColumnStats().getMin());
        numericInfo.setMean(columnConfig.getMean());
        numericInfo.setMedian(columnConfig.getMedian());
        numericInfo.setStandardDeviation(columnConfig.getStdDev());

        return numericInfo;
    }

    /**
     * Create @ConStats for numerical variable
     * 
     * @param columnConfig
     *            - ColumnConfig to generate ConStats
     * @return ConStats for variable
     */
    private ContStats createConStats(ColumnConfig columnConfig) {
        ContStats conStats = new ContStats();

        List<Interval> intervals = new ArrayList<Interval>();
        for(int i = 0; i < columnConfig.getBinBoundary().size(); i++) {
            Interval interval = new Interval();
            interval.setClosure(Interval.Closure.OPEN_CLOSED);
            interval.setLeftMargin(columnConfig.getBinBoundary().get(i));

            if(i == columnConfig.getBinBoundary().size() - 1) {
                interval.setRightMargin(Double.POSITIVE_INFINITY);
            } else {
                interval.setRightMargin(columnConfig.getBinBoundary().get(i + 1));
            }

            intervals.add(interval);
        }
        conStats.withIntervals(intervals);

        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", columnConfig.getBinCountPos().toString());
        extensionMap.put("BinCountNeg", columnConfig.getBinCountNeg().toString());
        extensionMap.put("BinWeightedCountPos", columnConfig.getBinWeightedPos().toString());
        extensionMap.put("BinWeightedCountNeg", columnConfig.getBinWeightedNeg().toString());
        extensionMap.put("BinPosRate", columnConfig.getBinPosRate().toString());
        extensionMap.put("BinWOE", calculateWoe(columnConfig.getBinCountPos(), columnConfig.getBinCountNeg())
                .toString());
        extensionMap.put("KS", Double.toString(columnConfig.getKs()));
        extensionMap.put("IV", Double.toString(columnConfig.getIv()));
        conStats.withExtensions(createExtensions(extensionMap));

        return conStats;
    }

    /**
     * Generate Woe data from positive and negative counts
     * 
     * @param binCountPos
     *            - positive count list
     * @param binCountNeg
     *            - negative count list
     * @return Woe value list
     */
    private List<Double> calculateWoe(List<Integer> binCountPos, List<Integer> binCountNeg) {
        List<Double> woe = new ArrayList<Double>();

        double sumPos = 0.0;
        double sumNeg = 0.0;

        for(int i = 0; i < binCountPos.size(); i++) {
            sumPos += binCountPos.get(i);
            sumNeg += binCountNeg.get(i);
        }

        for(int i = 0; i < binCountPos.size(); i++) {
            woe.add(Math.log((binCountPos.get(i) / sumPos + EPS) / (binCountNeg.get(i) / sumNeg + EPS)));
        }

        return woe;
    }
}
