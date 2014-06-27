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
package ml.shifu.core.core;

import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.container.obj.ColumnConfig.ColumnConfigComparator;
import ml.shifu.core.container.obj.ModelConfig;
import ml.shifu.core.core.alg.NNTrainer;
import ml.shifu.core.util.CommonUtils;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * variable selector
 */
public class VariableSelector {
    private static Logger log = LoggerFactory.getLogger(VariableSelector.class);

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;

    public VariableSelector(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        log.info("Creating VariableSelector...");
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }

    // return the list of selected column nums
    public List<ColumnConfig> selectByFilter() {
        log.info("    - Method: Filter");

        for (ColumnConfig config : this.columnConfigList) {
            config.setFinalSelect(false);
        }

        int ptrKs = 0, ptrIv = 0, cntByForce = 0;

        log.info("Start Variable Selection...");
        log.info("\t VarSelectEnabled: " + modelConfig.getVarSelectFilterEnabled());
        log.info("\t VarSelectBy: " + modelConfig.getVarSelectFilterBy());
        log.info("\t VarSelectNum: " + modelConfig.getVarSelectFilterNum());

        List<Integer> selectedColumnNumList = new ArrayList<Integer>();

        List<ColumnConfig> ksList = new ArrayList<ColumnConfig>();
        List<ColumnConfig> ivList = new ArrayList<ColumnConfig>();

        int cntSelected = 0;

        for (ColumnConfig config : this.columnConfigList) {
            if (config.isForceRemove()) {
                log.info("\t ForceRemove: " + config.getColumnName());
            } else if (config.isForceSelect()) {
                log.info("\t ForceSelect: " + config.getColumnName());
                if (config.getMean() == null || config.getStdDev() == null) {
                    // TODO - check the mean of categorical variable could be null
                    log.info("\t ForceSelect Failed: mean/stdDev must not be null");
                } else {
                    selectedColumnNumList.add(config.getColumnNum());
                    cntSelected++;
                    cntByForce++;
                }
            } else if (config.isMeta() || config.isTarget()) {
                log.info("\t Skip meta or target column: " + config.getColumnName());
            } else if (config.getColumnBinStatsResult().getKs() == null || config.getColumnBinStatsResult().getIv() == null) {
                log.info("\t Incomplete info: " + config.getColumnName());
            } else if ((config.isCategorical() && !modelConfig.isCategoricalDisabled()) || config.isNumerical()) {
                ksList.add(config);
                ivList.add(config);
            }
        }

        if (!this.modelConfig.getVarSelectFilterEnabled()) {
            log.info("Summary:");
            log.info("\tSelected Variables: " + cntSelected);

            if (cntByForce != 0) {
                log.info("\t- By Force: " + cntByForce);
            }

            for (int n : selectedColumnNumList) {
                this.columnConfigList.get(n).setFinalSelect(true);
            }

            return columnConfigList;
        }

        Collections.sort(ksList, new ColumnConfigComparator("ks"));
        Collections.sort(ivList, new ColumnConfigComparator("iv"));

        int expectedVarNum = Math.min(ksList.size(), this.modelConfig.getVarSelectFilterNum());

        String key = this.modelConfig.getVarSelectFilterBy();
        ColumnConfig config;
        while (cntSelected < expectedVarNum) {
            if (key.equalsIgnoreCase("ks")) {
                config = ksList.get(ptrKs);
                selectedColumnNumList.add(config.getColumnNum());
                ptrKs++;
                log.info("\t SelectedByKS=" + config.getColumnBinStatsResult().getKs() + "(Rank=" + ptrKs + "): " + config.getColumnName());

                cntSelected++;
            } else if (key.equalsIgnoreCase("iv")) {
                config = ivList.get(ptrIv);
                selectedColumnNumList.add(config.getColumnNum());
                ptrIv++;
                log.info("\t SelectedByIV=" + config.getColumnBinStatsResult().getIv() + "(Rank=" + ptrIv + "): " + config.getColumnName());

                cntSelected++;
            } else if (key.equalsIgnoreCase("mix")) {
                config = ksList.get(ptrKs);
                if (selectedColumnNumList.contains(config.getColumnNum())) {
                    log.info("\t Variable Already Selected: " + config.getColumnName());
                    ptrKs++;
                } else {
                    selectedColumnNumList.add(config.getColumnNum());
                    ptrKs++;
                    log.info("\t SelectedByKS=" + config.getColumnBinStatsResult().getKs() + "(Rank=" + ptrKs + "): " + config.getColumnName());
                    cntSelected++;
                }

                if (cntSelected == expectedVarNum) {
                    break;
                }

                config = ivList.get(ptrIv);
                if (selectedColumnNumList.contains(config.getColumnNum())) {
                    log.info("\t Variable Already Selected: " + config.getColumnName());
                    ptrIv++;
                } else {
                    selectedColumnNumList.add(config.getColumnNum());
                    ptrIv++;
                    log.info("\t SelectedByIV=" + config.getColumnBinStatsResult().getIv() + "(Rank=" + ptrIv + "): " + config.getColumnName());
                    cntSelected++;
                }
            }
        }

        log.info("Summary:");
        log.info("\t Selected Variables: " + cntSelected);
        if (cntByForce != 0) {
            log.info("\t - By Force: " + cntByForce);
        }
        if (ptrKs != 0) {
            log.info("\t - By KS: " + ptrKs);
        }
        if (ptrIv != 0) {
            log.info("\t - By IV: " + ptrIv);
        }

        for (int n : selectedColumnNumList) {
            this.columnConfigList.get(n).setFinalSelect(true);
        }

        return columnConfigList;
    }

    public List<ColumnConfig> selectByWrapper(NNTrainer trainer) {
        log.info("\t - Method: Wrapper");

        BasicNetwork network = trainer.getNetwork();
        Double baseMSE = trainer.getBaseMSE();
        MLDataSet validSet = trainer.getValidSet();

        List<ColumnConfig> candidateList = new ArrayList<ColumnConfig>();

        for (ColumnConfig config : columnConfigList) {
            if (config.isFinalSelect() == true) {
                candidateList.add(config);
            }
        }

        log.info("\t - Candidate ColumnConfig Size: " + candidateList.size());
        log.info("\t - Validation DataSet Size: " + validSet.getInputSize());
        if (candidateList.size() == 0) {
            throw new RuntimeException("No candidates. Run '$ core varselect -filter' first or manually set variables as finalSelect.");
        } else if (candidateList.size() != validSet.getInputSize()) {
            throw new RuntimeException("ColumnConfig and data mismatch.");
        }

        int size = candidateList.size();

        if (modelConfig.getVarSelectWrapperBy().equalsIgnoreCase("A")) {
            log.info("\t - By Adding Most Significant Variables");
            int iterations = modelConfig.getVarSelectWrapperNum();
            Set<Integer> selected = new HashSet<Integer>();

            for (int n = 1; n <= iterations; n++) {
                double maxDiffMSE = Double.NEGATIVE_INFINITY;
                int maxDiffMSEColumn = -1;

                log.info("\t Iteration #" + n);

                for (int i = 0; i < size; i++) {
                    if (selected.contains(i)) {
                        continue;
                    }

                    Double mse = getMSE(network, validSet, selected, i);
                    if (mse - baseMSE > maxDiffMSE) {
                        maxDiffMSE = mse - baseMSE;
                        maxDiffMSEColumn = i;
                    }
                }

                selected.add(maxDiffMSEColumn);
                log.info("\t Selected Variable: " + candidateList.get(maxDiffMSEColumn).getColumnName());
                log.info("\t MSE: " + maxDiffMSE);
            }

            for (ColumnConfig config : columnConfigList) {
                config.setFinalSelect(false);
            }

            for (Integer i : selected) {
                columnConfigList.get(candidateList.get(i).getColumnNum()).setFinalSelect(true);
            }

        } else if (modelConfig.getVarSelectWrapperBy().equalsIgnoreCase("R")) {
            log.info("\t - By Removing Least Significant Variables");
            int iterations = candidateList.size() - modelConfig.getVarSelectWrapperNum();
            Set<Integer> removed = new HashSet<Integer>();

            for (int n = 1; n <= iterations; n++) {
                double minDiffMSE = Double.POSITIVE_INFINITY;

                int minDiffMSEColumn = -1;

                log.info("\t Iteration #" + n);

                for (int i = 0; i < size; i++) {
                    if (removed.contains(i)) {
                        continue;
                    }

                    Double mse = getMSE(network, validSet, removed, i);
                    if (Math.abs(mse - baseMSE) < minDiffMSE) {
                        minDiffMSE = Math.abs(mse - baseMSE);
                        minDiffMSEColumn = i;
                    }
                }

                removed.add(minDiffMSEColumn);
                log.info("\t Removed: Variable: " + candidateList.get(minDiffMSEColumn).getColumnName());
                log.info("\t MSE: " + minDiffMSE);
            }
            for (Integer i : removed) {
                columnConfigList.get(candidateList.get(i).getColumnNum()).setFinalSelect(false);
            }

        } else if (modelConfig.getVarSelectWrapperBy().equalsIgnoreCase("S")) {
            log.info("\t - Simplified Wrapper Method");

            Map<Integer, Double> mseMap = new HashMap<Integer, Double>();

            for (int i = 0; i < size; i++) {
                mseMap.put(i, getMSE(network, validSet, null, i));
            }

            List<Map.Entry<Integer, Double>> entryList = CommonUtils.getEntriesSortedByValues(mseMap);
            int numVars = modelConfig.getVarSelectWrapperNum();

            for (ColumnConfig config : columnConfigList) {
                config.setFinalSelect(false);
            }

            int entryListSize = entryList.size();
            if (numVars > entryListSize) {
                numVars = entryListSize;
            }

            for (int i = 0; i < numVars; i++) {
                Map.Entry<Integer, Double> entry = entryList.get(entryListSize - i - 1);
                ColumnConfig config = candidateList.get(entry.getKey());
                log.info(config.getColumnName() + ": " + entry.getValue());
                columnConfigList.get(config.getColumnNum()).setFinalSelect(true);
            }
        } else {
            log.error("Invalid Wrapper Method. Choose from 'A', 'R' or 'S'");
        }

        return columnConfigList;
    }

    private Double getMSE(BasicNetwork network, MLDataSet validSet, Set<Integer> clamped, Integer clamping) {
        MLDataSet tmpSet = new BasicMLDataSet();
        for (MLDataPair validPair : validSet) {
            // Make a copy of validPair
            double[] input = validPair.getInputArray().clone();
            double[] ideal = validPair.getIdealArray().clone();

            // Set one variable to mean(normalized);
            input[clamping] = 0;

            // Set selected variables to mean;
            if (clamped != null) {
                for (Integer k : clamped) {
                    input[k] = 0;
                }
            }

            // Add to tmp Set
            tmpSet.add(new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal)));
        }

        return AbstractTrainer.calculateMSE(network, tmpSet);
    }
}
