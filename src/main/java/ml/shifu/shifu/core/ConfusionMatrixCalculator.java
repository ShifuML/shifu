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

import ml.shifu.shifu.container.ConfusionMatrixObject;
import ml.shifu.shifu.container.ModelResultObject;
import ml.shifu.shifu.util.QuickSort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Confusion matrix calculator
 */
public class ConfusionMatrixCalculator {

    private static Logger log = LoggerFactory.getLogger(ConfusionMatrixCalculator.class);

    // input
    private List<ModelResultObject> moList;
    private List<String> posTags;

    @SuppressWarnings("unused")
    private List<String> negTags;

    private Double negScaleFactor = 1.0;
    private Double posScaleFactor = 1.0;

    private static String fmt = "%s|%s|%s|%s|%s|%s|%s|%s|%s\n";

    public ConfusionMatrixCalculator(List<String> posTags, List<String> negTags, List<ModelResultObject> moList) {
        this.moList = moList;
        this.posTags = posTags;
        this.negTags = negTags;
        QuickSort.sort(this.moList, new ModelResultObject.ModelResultObjectComparator());
    }

    public List<ConfusionMatrixObject> calculate() {

        List<ConfusionMatrixObject> cmoList = new ArrayList<ConfusionMatrixObject>();

        // Calculate the sum
        Double sumPos = 0.0, sumNeg = 0.0, sumWeightedPos = 0.0, sumWeightedNeg = 0.0;
        for(ModelResultObject mo: moList) {
            if(posTags.contains(mo.getTag())) {
                // Positive
                sumPos += posScaleFactor;
                sumWeightedPos += mo.getWeight() * posScaleFactor;
            } else {
                // Negative
                sumNeg += negScaleFactor;
                sumWeightedNeg += mo.getWeight() * negScaleFactor;
            }
        }

        // init ConfusionMatrix
        ConfusionMatrixObject initCmo = new ConfusionMatrixObject();
        initCmo.setTp(0.0);
        initCmo.setFp(0.0);
        initCmo.setFn(sumPos);
        initCmo.setTn(sumNeg);
        initCmo.setWeightedTp(0.0);
        initCmo.setWeightedFp(0.0);
        initCmo.setWeightedFn(sumWeightedPos);
        initCmo.setWeightedTn(sumWeightedNeg);
        initCmo.setScore(moList.get(0).getScore());
        cmoList.add(initCmo);

        // Calculate the rest
        ConfusionMatrixObject prevCmo = initCmo;
        for(ModelResultObject mo: moList) {
            ConfusionMatrixObject cmo = new ConfusionMatrixObject(prevCmo);

            if(posTags.contains(mo.getTag())) {
                // Positive Instance
                cmo.setTp(cmo.getTp() + posScaleFactor);
                cmo.setFn(cmo.getFn() - posScaleFactor);
                cmo.setWeightedTp(cmo.getWeightedTp() + mo.getWeight() * posScaleFactor);
                cmo.setWeightedFn(cmo.getWeightedFn() - mo.getWeight() * posScaleFactor);
            } else {
                // Negative Instance
                cmo.setFp(cmo.getFp() + negScaleFactor);
                cmo.setTn(cmo.getTn() - negScaleFactor);
                cmo.setWeightedFp(cmo.getWeightedFp() + mo.getWeight() * negScaleFactor);
                cmo.setWeightedTn(cmo.getWeightedTn() - mo.getWeight() * negScaleFactor);
            }

            cmo.setScore(mo.getScore());
            cmoList.add(cmo);
            prevCmo = cmo;
        }

        return cmoList;
    }

    public void setNegScaleFactor(Double negScaleFactor) {
        this.negScaleFactor = negScaleFactor;
    }

    public void setPosScaleFactor(Double posScaleFactor) {
        this.posScaleFactor = posScaleFactor;
    }

    public void calculate(BufferedWriter writer) {

        Double sumPos = 0.0, sumNeg = 0.0, sumWeightedPos = 0.0, sumWeightedNeg = 0.0;
        for(ModelResultObject mo: moList) {
            if(posTags.contains(mo.getTag())) {
                // Positive
                sumPos += posScaleFactor;
                sumWeightedPos += mo.getWeight() * posScaleFactor;
            } else {
                // Negative
                sumNeg += negScaleFactor;
                sumWeightedNeg += mo.getWeight() * negScaleFactor;
            }
        }

        ConfusionMatrixObject prevCmo = new ConfusionMatrixObject();

        prevCmo.setTp(0.0);
        prevCmo.setFp(0.0);
        prevCmo.setFn(sumPos);
        prevCmo.setTn(sumNeg);
        prevCmo.setWeightedTp(0.0);
        prevCmo.setWeightedFp(0.0);
        prevCmo.setWeightedFn(sumWeightedPos);
        prevCmo.setWeightedTn(sumWeightedNeg);
        prevCmo.setScore(1000);

        saveConfusionMaxtrixWithWriter(writer, prevCmo);

        for(ModelResultObject mo: moList) {
            ConfusionMatrixObject cmo = new ConfusionMatrixObject(prevCmo);

            if(posTags.contains(mo.getTag())) {
                // Positive Instance
                cmo.setTp(cmo.getTp() + posScaleFactor);
                cmo.setFn(cmo.getFn() - posScaleFactor);
                cmo.setWeightedTp(cmo.getWeightedTp() + mo.getWeight() * posScaleFactor);
                cmo.setWeightedFn(cmo.getWeightedFn() - mo.getWeight() * posScaleFactor);
            } else {
                // Negative Instance
                cmo.setFp(cmo.getFp() + negScaleFactor);
                cmo.setTn(cmo.getTn() - negScaleFactor);
                cmo.setWeightedFp(cmo.getWeightedFp() + mo.getWeight() * negScaleFactor);
                cmo.setWeightedTn(cmo.getWeightedTn() - mo.getWeight() * negScaleFactor);
            }

            cmo.setScore(mo.getScore());
            saveConfusionMaxtrixWithWriter(writer, cmo);
            prevCmo = cmo;
        }

    }

    public static void saveConfusionMaxtrixWithWriter(BufferedWriter writer, ConfusionMatrixObject cmo) {
        try {
            writer.write(String.format(fmt, cmo.getTp(), cmo.getFp(), cmo.getFn(), cmo.getTn(), cmo.getWeightedTp(),
                    cmo.getWeightedFp(), cmo.getWeightedFn(), cmo.getWeightedTn(), cmo.getScore()));
        } catch (IOException e) {
            try {
                writer.close();
            } catch (IOException e1) {
                log.error("Could not close the writer while write into confusion matrix");
            }

            log.error("Could not write into confusion matrix");
        }

    }

}
