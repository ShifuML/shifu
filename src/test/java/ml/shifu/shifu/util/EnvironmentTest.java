/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.util;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

/**
 * EnvironmentTest class
 */
public class EnvironmentTest {

    @Test
    public void testLoadShifuConfig() throws IOException {
        Environment.setProperty("SHIFU_HOME", "src/main/resources");
        Environment.loadShifuConfig();
        Assert.assertEquals(6, Environment.getInt(Environment.LOCAL_NUM_PARALLEL).intValue());
        Assert.assertNotNull(Environment.getProperty(Environment.SYSTEM_USER));
    }
}
