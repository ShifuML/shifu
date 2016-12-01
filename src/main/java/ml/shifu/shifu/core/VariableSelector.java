/**
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

import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnConfigComparator;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.alg.NNTrainer;
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

    // return the list of selected column nums
    public List<ColumnConfig> selectByFilter() {
        log.info("    - Method: Filter");

        for(ColumnConfig config: this.columnConfigList) {
            config.setFinalSelect(false);
        }

        int ptrKs = 0, ptrIv = 0, ptrPareto = 0, cntByForce = 0;

        log.info("Start Variable Selection...");
        log.info("\t VarSelectEnabled: " + modelConfig.getVarSelectFilterEnabled());
        log.info("\t VarSelectBy: " + modelConfig.getVarSelectFilterBy());
        log.info("\t VarSelectNum: " + modelConfig.getVarSelectFilterNum());

        List<Integer> selectedColumnNumList = new ArrayList<Integer>();

        List<ColumnConfig> ksList = new ArrayList<ColumnConfig>();
        List<ColumnConfig> ivList = new ArrayList<ColumnConfig>();
        List<Tuple> paretoList = new ArrayList<Tuple>();

        int cntSelected = 0;

        for(ColumnConfig config: this.columnConfigList) {
            if(config.isForceRemove()) {
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
            } else if(config.isMeta() || config.isTarget()) {
                log.info("\t Skip meta or target column: " + config.getColumnName());
            } else if(!CommonUtils.isGoodCandidate(config)) {
                log.info("\t Incomplete info(please check KS, IV, Mean, or StdDev fields): " + config.getColumnName());
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

        int expectedVarNum = Math.min(ksList.size(), this.modelConfig.getVarSelectFilterNum());

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
                    // System.out.println("remove 1");
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

    /**
     * TODO
     */
    public List<Tuple> sortByPareto(List<Tuple> paretoList) {
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

    public List<ColumnConfig> selectByWrapper(NNTrainer trainer) {
        log.info("\t - Method: Wrapper");

        BasicNetwork network = trainer.getNetwork();
        Double baseMSE = trainer.getBaseMSE();
        MLDataSet validSet = trainer.getValidSet();

        List<ColumnConfig> candidateList = new ArrayList<ColumnConfig>();

        for(ColumnConfig config: columnConfigList) {
            if(config.isFinalSelect() == true) {
                candidateList.add(config);
            }
        }

        log.info("\t - Candidate ColumnConfig Size: " + candidateList.size());
        log.info("\t - Validation DataSet Size: " + validSet.getInputSize());
        if(candidateList.size() == 0) {
            throw new RuntimeException(
                    "No candidates. Run '$ shifu varselect -filter' first or manually set variables as finalSelect.");
        } else if(candidateList.size() != validSet.getInputSize()) {
            throw new RuntimeException("ColumnConfig and data mismatch.");
        }

        int size = candidateList.size();

        if(modelConfig.getVarSelectWrapperBy().equalsIgnoreCase("A")) {
            log.info("\t - By Adding Most Significant Variables");
            int iterations = modelConfig.getVarSelectWrapperNum();
            Set<Integer> selected = new HashSet<Integer>();

            for(int n = 1; n <= iterations; n++) {
                double maxDiffMSE = Double.NEGATIVE_INFINITY;
                int maxDiffMSEColumn = -1;

                log.info("\t Iteration #" + n);

                for(int i = 0; i < size; i++) {
                    if(selected.contains(i)) {
                        continue;
                    }

                    Double mse = getMSE(network, validSet, selected, i);
                    if(mse - baseMSE > maxDiffMSE) {
                        maxDiffMSE = mse - baseMSE;
                        maxDiffMSEColumn = i;
                    }
                }

                selected.add(maxDiffMSEColumn);
                log.info("\t Selected Variable: " + candidateList.get(maxDiffMSEColumn).getColumnName());
                log.info("\t MSE: " + maxDiffMSE);
            }

            for(ColumnConfig config: columnConfigList) {
                config.setFinalSelect(false);
            }

            for(Integer i: selected) {
                columnConfigList.get(candidateList.get(i).getColumnNum()).setFinalSelect(true);
            }

        } else if(modelConfig.getVarSelectWrapperBy().equalsIgnoreCase("R")) {
            log.info("\t - By Removing Least Significant Variables");
            int iterations = candidateList.size() - modelConfig.getVarSelectWrapperNum();
            Set<Integer> removed = new HashSet<Integer>();

            for(int n = 1; n <= iterations; n++) {
                double minDiffMSE = Double.POSITIVE_INFINITY;

                int minDiffMSEColumn = -1;

                log.info("\t Iteration #" + n);

                for(int i = 0; i < size; i++) {
                    if(removed.contains(i)) {
                        continue;
                    }

                    Double mse = getMSE(network, validSet, removed, i);
                    if(Math.abs(mse - baseMSE) < minDiffMSE) {
                        minDiffMSE = Math.abs(mse - baseMSE);
                        minDiffMSEColumn = i;
                    }
                }

                removed.add(minDiffMSEColumn);
                log.info("\t Removed: Variable: " + candidateList.get(minDiffMSEColumn).getColumnName());
                log.info("\t MSE: " + minDiffMSE);
            }
            for(Integer i: removed) {
                columnConfigList.get(candidateList.get(i).getColumnNum()).setFinalSelect(false);
            }

        } else if(modelConfig.getVarSelectWrapperBy().equalsIgnoreCase("S")) {
            log.info("\t - Simplified Wrapper Method");

            Map<Integer, Double> mseMap = new HashMap<Integer, Double>();

            for(int i = 0; i < size; i++) {
                mseMap.put(i, getMSE(network, validSet, null, i));
            }

            List<Map.Entry<Integer, Double>> entryList = CommonUtils.getEntriesSortedByValues(mseMap);
            int numVars = modelConfig.getVarSelectWrapperNum();

            for(ColumnConfig config: columnConfigList) {
                config.setFinalSelect(false);
            }

            int entryListSize = entryList.size();
            if(numVars > entryListSize) {
                numVars = entryListSize;
            }

            for(int i = 0; i < numVars; i++) {
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
        for(MLDataPair validPair: validSet) {
            // Make a copy of validPair
            double[] input = validPair.getInputArray().clone();
            double[] ideal = validPair.getIdealArray().clone();

            // Set one variable to mean(normalized);
            input[clamping] = 0;

            // Set selected variables to mean;
            if(clamped != null) {
                for(Integer k: clamped) {
                    input[k] = 0;
                }
            }

            // Add to tmp Set
            tmpSet.add(new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal)));
        }

        return AbstractTrainer.calculateMSE(network, tmpSet);
    }
}
