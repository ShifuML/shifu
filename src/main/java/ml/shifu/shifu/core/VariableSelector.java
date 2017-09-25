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

import java.io.IOException;
import java.util.*;

import ml.shifu.shifu.column.NSColumn;
import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnConfigComparator;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;

/**
 * variable selector
 */
public class VariableSelector {
    private static Logger log = LoggerFactory.getLogger(VariableSelector.class);

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;

    private double[] epsilonArray = new double[] { 0.01d, 0.05d };

    public VariableSelector(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        log.info("Creating VariableSelector...");
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.epsilonArray = modelConfig.getVarSelect().getEpsilons();
    }

    // TODO it should support some DSL like "KS > 2 and IV and PSI <= 0.1"
    public List<ColumnConfig> selectByFilter(String input) {
        // VarSelParser parser = new VarSelParser(new CommonTokenStream(new VarSelLexer(new ANTLRInputStream(input))));
        return null;
    }

    public static class Tuple {

        public int columnNum;

        public double ks;

        public double iv;

        public double[] box;

        public Tuple(int columnNum, double ks, double iv) {
            this.columnNum = columnNum;
            this.ks = ks;
            this.iv = iv;
        }

        public static Tuple clone(Tuple tuple) {
            Tuple newOne = new Tuple(tuple.columnNum, tuple.ks, tuple.iv);
            if(tuple.box != null) {
                double[] newBox = new double[tuple.box.length];
                for(int i = 0; i < newBox.length; i++) {
                    newBox[i] = tuple.box[i];
                }
                newOne.box = newBox;
            }
            return newOne;
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#toString()
         */
        @Override
        public String toString() {
            return "Tuple [columnNum=" + columnNum + ", ks=" + ks + ", iv=" + iv + ", box=" + Arrays.toString(box)
                    + "]";
        }
    }

    public static void setFilterNumberByFilterOutRatio(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        // if user already set filter number then ignore filter out ratio
        if(modelConfig.getVarSelectFilterNum() > 0) {
            return;
        }
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(), columnConfigList);
        int inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        Float filterOutRatio = modelConfig.getVarSelect().getFilterOutRatio();
        int targetCnt = (int) (inputNodeCount * (1.0f - filterOutRatio));
        modelConfig.getVarSelect().setFilterNum(targetCnt);
    }

    // return the list of selected column nums
    public List<ColumnConfig> selectByFilter() throws IOException {
        log.info("    - Method: Filter");

        int ptrKs = 0, ptrIv = 0, ptrPareto = 0, cntByForce = 0;
        VariableSelector.setFilterNumberByFilterOutRatio(this.modelConfig, this.columnConfigList);
        log.info("Start Variable Selection...");
        log.info("\t VarSelectEnabled: " + modelConfig.getVarSelectFilterEnabled());
        log.info("\t VarSelectBy: " + modelConfig.getVarSelectFilterBy());
        log.info("\t VarSelectNum: " + modelConfig.getVarSelectFilterNum());

        List<Integer> selectedColumnNumList = new ArrayList<Integer>();

        List<ColumnConfig> ksList = new ArrayList<ColumnConfig>();
        List<ColumnConfig> ivList = new ArrayList<ColumnConfig>();
        List<Tuple> paretoList = new ArrayList<Tuple>();

        Set<NSColumn> candidateColumns = CommonUtils.loadCandidateColumns(modelConfig);
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        int cntSelected = 0;

        for(ColumnConfig config: this.columnConfigList) {
            if(config.isMeta() || config.isTarget()) {
                log.info("\t Skip meta, weight or target column: " + config.getColumnName());
            } else if(config.isForceRemove()) {
                log.info("\t ForceRemove: " + config.getColumnName());
            } else if(config.isForceSelect()) {
                log.info("\t ForceSelect: " + config.getColumnName());
                if(config.getMean() == null || config.getStdDev() == null) {
                    // TODO - check the mean of categorical variable could be null
                    log.info("\t ForceSelect Failed: mean/stdDev must not be null");
                } else {
                    selectedColumnNumList.add(config.getColumnNum());
                    cntSelected++;
                    cntByForce++;
                }
            } else if(!CommonUtils.isGoodCandidate(config, hasCandidates)) {
                log.info("\t Incomplete info(please check KS, IV, Mean, or StdDev fields): " + config.getColumnName()
                    + " or it is not in candidate list");
            } else if (CollectionUtils.isNotEmpty(candidateColumns)
                    && !candidateColumns.contains(new NSColumn(config.getColumnName()))) {
                log.info("\t Not in candidate list, Skip: " + config.getColumnName());
            } else if((config.isCategorical() && !modelConfig.isCategoricalDisabled()) || config.isNumerical()) {
                ksList.add(config);
                ivList.add(config);
                if(config != null && config.getColumnStats() != null) {
                    Double ks = config.getKs();
                    Double iv = config.getIv();
                    paretoList.add(new Tuple(config.getColumnNum(), ks == null ? 0d : ks, iv == null ? 0d : iv));
                }
            }
        }

        // not enabled filter, so only select forceSelect columns
        if(!this.modelConfig.getVarSelectFilterEnabled()) {
            log.info("Summary:");
            log.info("\tSelected Variables: " + cntSelected);

            if(cntByForce != 0) {
                log.info("\t- By Force: " + cntByForce);
            }

            for(int n: selectedColumnNumList) {
                this.columnConfigList.get(n).setFinalSelect(true);
            }

            return columnConfigList;
        }

        String key = this.modelConfig.getVarSelectFilterBy();

        Collections.sort(ksList, new ColumnConfigComparator("ks"));
        Collections.sort(ivList, new ColumnConfigComparator("iv"));

        List<Tuple> newParetoList = sortByPareto(paretoList);

        int expectedVarNum = Math.min(cntSelected + ksList.size(), modelConfig.getVarSelectFilterNum());
        log.info("Expected selected columns:" + expectedVarNum);

        // reset to false at first.
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                columnConfig.setFinalSelect(false);
            }
        }
        ColumnConfig config = null;
        while(cntSelected < expectedVarNum) {
            if(key.equalsIgnoreCase("ks")) {
                config = ksList.get(ptrKs);
                selectedColumnNumList.add(config.getColumnNum());
                ptrKs++;
                log.info("\t SelectedByKS=" + config.getKs() + "(Rank=" + ptrKs + "): " + config.getColumnName());

                cntSelected++;
            } else if(key.equalsIgnoreCase("iv")) {
                config = ivList.get(ptrIv);
                selectedColumnNumList.add(config.getColumnNum());
                ptrIv++;
                log.info("\t SelectedByIV=" + config.getIv() + "(Rank=" + ptrIv + "): " + config.getColumnName());

                cntSelected++;
            } else if(key.equalsIgnoreCase("mix")) {
                config = ksList.get(ptrKs);
                if(selectedColumnNumList.contains(config.getColumnNum())) {
                    log.info("\t Variable Already Selected: " + config.getColumnName());
                    ptrKs++;
                } else {
                    selectedColumnNumList.add(config.getColumnNum());
                    ptrKs++;
                    log.info("\t SelectedByKS=" + config.getKs() + "(Rank=" + ptrKs + "): " + config.getColumnName());
                    cntSelected++;
                }

                if(cntSelected == expectedVarNum) {
                    break;
                }

                config = ivList.get(ptrIv);
                if(selectedColumnNumList.contains(config.getColumnNum())) {
                    log.info("\t Variable Already Selected: " + config.getColumnName());
                    ptrIv++;
                } else {
                    selectedColumnNumList.add(config.getColumnNum());
                    ptrIv++;
                    log.info("\t SelectedByIV=" + config.getIv() + "(Rank=" + ptrIv + "): " + config.getColumnName());
                    cntSelected++;
                }
            } else if(key.equalsIgnoreCase("pareto")) {
                if(ptrPareto >= newParetoList.size()) {
                    config = ksList.get(ptrKs);
                    if(selectedColumnNumList.contains(config.getColumnNum())) {
                        log.info("\t Variable Already Selected: " + config.getColumnName());
                    } else {
                        selectedColumnNumList.add(config.getColumnNum());
                        log.info("\t SelectedByKS=" + config.getKs() + "(Rank=" + ptrKs + newParetoList.size() + "): "
                                + config.getColumnName());
                        cntSelected++;
                    }
                    ptrKs++;
                } else {
                    int columnNum = newParetoList.get(ptrPareto).columnNum;
                    selectedColumnNumList.add(columnNum);
                    log.info("\t SelectedByPareto " + columnConfigList.get(columnNum).getColumnName());
                    ptrPareto++;
                    cntSelected++;
                }
            }
        }

        log.info("Summary:");
        log.info("\t Selected Variables: " + cntSelected);
        if(cntByForce != 0) {
            log.info("\t - By Force: " + cntByForce);
        }

        if(ptrPareto != 0) {
            log.info("\t - By Pareto: " + ptrPareto);
        }

        if(ptrKs != 0) {
            log.info("\t - By KS: " + ptrKs);
        }
        if(ptrIv != 0) {
            log.info("\t - By IV: " + ptrIv);
        }

        // update column config list and set finalSelect to true
        for(int n: selectedColumnNumList) {
            this.columnConfigList.get(n).setFinalSelect(true);
        }

        return columnConfigList;
    }

    private static class Archives {

        public double[] epsilons;

        public List<Tuple> tuples = new ArrayList<VariableSelector.Tuple>();

        public Archives(double[] epsilons) {
            this.epsilons = epsilons;
        }

        public void sortInto(Tuple currTuple) {
            double[] eBox = new double[epsilons.length];
            for(int i = 0; i < epsilons.length; i++) {
                if(i == 0) {
                    eBox[i] = Math.floor(currTuple.ks / epsilons[i]);
                } else {
                    eBox[i] = Math.floor(currTuple.iv / epsilons[i]);
                }
            }
            // System.out.println(Arrays.toString(eBox));
            int currSize = tuples.size();
            int index = -1;

            while(index < currSize - 1) {
                index += 1;
                boolean adominate = false; // # archive dominates
                boolean sdominate = false; // # solution dominates
                boolean nondominate = false; // # neither dominates

                Tuple indexTuple = tuples.get(index);
                double[] aBox = indexTuple.box;
                // System.out.println(Arrays.toString(aBox));
                for(int i = 0; i < epsilons.length; i++) {
                    if(aBox[i] < eBox[i]) {
                        adominate = true;
                        if(sdominate) {
                            nondominate = true;
                            break; // for;
                        }
                    } else if(aBox[i] > eBox[i]) {
                        sdominate = true;
                        if(adominate) { // # nondomination
                            nondominate = true;
                            break;// # for
                        }
                    }
                }

                if(nondominate) {
                    continue;
                }// # while
                if(adominate) {// # candidate solution was dominated
                    return;
                }
                if(sdominate) { // # candidate solution dominated archive solution
                    // System.out.println(currTuple.columnNum + " " + currTuple.ks + " " + currTuple.iv);
                    // System.out.println(index);
                    this.tuples.remove(index);
                    index -= 1;
                    currSize -= 1;
                    continue; // # while
                }
                // # solutions are in the same box
                indexTuple = tuples.get(index);
                // corner = [ebox[ii] * self.epsilons[ii] for ii in self.itobj]
                double[] corner = new double[epsilons.length];
                for(int j = 0; j < corner.length; j++) {
                    corner[j] = eBox[j] * epsilons[j];
                }

                double sdist = 0d, adist = 0d;
                for(int j = 0; j < corner.length; j++) {
                    if(j == 0) {
                        sdist += (currTuple.ks - corner[j]) * (currTuple.ks - corner[j]);
                        adist += (indexTuple.ks - corner[j]) * (indexTuple.ks - corner[j]);
                    } else {
                        sdist += (currTuple.iv - corner[j]) * (currTuple.iv - corner[j]);
                        adist += (indexTuple.iv - corner[j]) * (indexTuple.iv - corner[j]);
                    }
                }

                if(adist < sdist) {// # archive dominates
                    return;
                } else { // : # solution dominates
                    this.tuples.remove(index);
                    index -= 1;
                    currSize -= 1;
                    continue; // # while
                }
            }

            // if you get here, then no archive solution has dominated this one
            currTuple.box = eBox;
            tuples.add(Tuple.clone(currTuple));
        }
    }

    public List<Tuple> sortByPareto(List<Tuple> paretoList) {
        // TODO
        if(this.epsilonArray == null) {
            this.epsilonArray = new double[] { 0.01d, 0.05d };
        }

        Archives ar = new Archives(this.epsilonArray);
        for(Tuple tuple: paretoList) {
            ar.sortInto(tuple);
        }
        return ar.tuples;
    }

    public List<Tuple> sortByParetoCC(List<ColumnConfig> list) {
        if(this.epsilonArray == null) {
            this.epsilonArray = new double[] { 0.01d, 0.05d };
        }

        List<Tuple> tuples = new ArrayList<VariableSelector.Tuple>();
        for(ColumnConfig columnConfig: list) {
            if(columnConfig != null && columnConfig.getColumnStats() != null) {
                Double ks = columnConfig.getKs();
                Double iv = columnConfig.getIv();
                tuples.add(new Tuple(columnConfig.getColumnNum(), ks == null ? 0d : ks, iv == null ? 0d : 0 - iv));
            }
        }
        return sortByPareto(tuples);
    }

}
