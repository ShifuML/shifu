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
package ml.shifu.shifu.util;


import ml.shifu.shifu.core.BasicStatsCalculator;
import org.easymock.EasyMock;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;


public class BasicStatsCalculatorTest {

    private BasicStatsCalculator mock;

    @BeforeClass
    public void setUp() {
        mock = EasyMock.createNiceMock(BasicStatsCalculator.class);
        EasyMock.expect(mock.getMax()).andReturn(1000.0).times(2);
        EasyMock.replay(mock);
    }

    @Test
    public void testGetMax() {
        Assert.assertTrue(mock.getMax() > 999.9);
        Assert.assertTrue(mock.getMax() < 1000.1);
        mock.getMin();
    }

    @Test
    public void testAnswer() {
        List<?> list = EasyMock.createMock(List.class);


        // andDelegateTo style
        EasyMock.expect(list.remove(10)).andDelegateTo(new ArrayList<String>() {
            private static final long serialVersionUID = 6489907692285763374L;

            @Override
            public String remove(int index) {
                return Integer.toString(index);
            }
        });

        EasyMock.replay(list);

        Assert.assertEquals(list.remove(10), "10");
    }

}
