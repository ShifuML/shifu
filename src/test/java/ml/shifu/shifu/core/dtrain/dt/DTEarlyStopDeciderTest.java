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
package ml.shifu.shifu.core.dtrain.dt;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * Created by haifwu on 2016/12/16.
 */
public class DTEarlyStopDeciderTest {
    private static final Logger LOG = LoggerFactory.getLogger(DTEarlyStopDeciderTest.class);

    private List<Double> trainErrorList = new ArrayList<Double>();
    private List<Double> validationErrorList = new ArrayList<Double>();
    private final static String DATA_FILE_NAME = "dttest/data/trainError.dat";

    @SuppressWarnings("unused")
    @BeforeMethod
    public void setUp() throws Exception {
        BasicConfigurator.configure();
        LogManager.getRootLogger().setLevel(Level.DEBUG);
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(this.getClass().getClassLoader().getResource(DATA_FILE_NAME).getFile());

        Scanner scanner = new Scanner(file);
        while(scanner.hasNext()) {
            String line = scanner.nextLine();
            String[] info = line.split("\\t");
            if(info.length != 2) {
                LOG.error("Wrong format of line: " + line);
            }
            trainErrorList.add(Double.valueOf(info[0]));
            validationErrorList.add(Double.valueOf(info[1]));
        }
        scanner.close();
    }

    @Test
    public void testAdd() throws Exception {
        DTEarlyStopDecider dtEarlyStopDecider = new DTEarlyStopDecider(6);
        Assert.assertEquals(trainErrorList.size(), validationErrorList.size());

        int iteration = 0;
        while(iteration++ < trainErrorList.size()) {
            if(dtEarlyStopDecider.add(trainErrorList.get(iteration), validationErrorList.get(iteration))) {
                LOG.info("Iteration {} stop!", iteration);
                break;
            }
        }

        Assert.assertNotSame(iteration, trainErrorList.size());
    }

}