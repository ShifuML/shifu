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

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * @author Wu Devin (wuhaifengdhu@163.com)
 */
public class DTEarlyStopDeciderTest {
    private static final Logger LOG = LoggerFactory.getLogger(DTEarlyStopDeciderTest.class);

    private List<Double> validationErrorList = new ArrayList<Double>();
    private final static String DATA_FILE_NAME = "dttest/data/trainError.dat";

    @BeforeMethod
    public void setUp() throws Exception {
        BasicConfigurator.configure();
        LogManager.getRootLogger().setLevel(Level.DEBUG);
        URL resource = this.getClass().getClassLoader().getResource(DATA_FILE_NAME);
        if(resource != null) {
            File file = new File(resource.getFile());
            Scanner scanner = new Scanner(file);
            while(scanner.hasNext()) {
                String line = scanner.nextLine();
                this.validationErrorList.add(Double.valueOf(line));
            }
            scanner.close();
        } else {
            LOG.error("Resource file {} not exist!", DATA_FILE_NAME);
        }
    }

    @Test
    public void testAdd() throws Exception {
        DTEarlyStopDecider dtEarlyStopDecider = new DTEarlyStopDecider(6);
        LOG.info("Total iteration size: {}", this.validationErrorList.size());

        int iteration = 0;
        for(; iteration < this.validationErrorList.size(); iteration++) {
            if(dtEarlyStopDecider.add(this.validationErrorList.get(iteration))) {
                // LOG.info("Iteration {} stop!", iteration);
                break;
            }
        }

        Assert.assertNotSame(iteration, this.validationErrorList.size());
    }

    @Test
    public void testGetCurrentAverageValue() {
        DTEarlyStopDecider dtEarlyStopDecider = new DTEarlyStopDecider(6);
        LOG.info("Total iteration size: {}", this.validationErrorList.size());

        int iteration = 0;
        for(; iteration < this.validationErrorList.size(); iteration++) {
            if(dtEarlyStopDecider.add(this.validationErrorList.get(iteration))) {
                LOG.info("Iteration {} stop!", iteration);
                break;
            }
            // LOG.info("iteration {}: {}==> average value {}", iteration, this.validationErrorList.get(iteration),
            // dtEarlyStopDecider.getCurrentAverageValue());
        }

        Assert.assertNotSame(iteration, this.validationErrorList.size());
    }
}