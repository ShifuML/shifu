/*
 * Copyright [2013-2015] PayPal Software Foundation
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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.*;

public class QuickKFindTest {

    public Comparator<Integer> comparator = new Comparator<Integer>() {

        @Override
        public int compare(Integer o1, Integer o2) {
            return o1.compareTo(o2);
        }
    };


    @Test
    public void test() {
        List<Integer> objList = null;
        Assert.assertNull(FindKValue.find(objList, 10, comparator));

        List<Integer> dataList = new ArrayList<Integer>();

        Random r = new Random((new Date()).getTime());

        dataList.add(r.nextInt(20));
        dataList.add(r.nextInt(20));
        dataList.add(r.nextInt(20));
        dataList.add(r.nextInt(20));
        dataList.add(r.nextInt(20));
        dataList.add(r.nextInt(20));

        //System.out.println(dataList.toString());

        List<Integer> copy_of_dataList = new ArrayList<Integer>(dataList);
        Collections.copy(copy_of_dataList, dataList);
        Collections.sort(copy_of_dataList);

        int i = r.nextInt(dataList.size());

        Integer rt = copy_of_dataList.get(i);

        Assert.assertEquals(FindKValue.find(dataList, i, comparator), rt);

        //System.out.println(copy_of_dataList.toString());
        //System.out.println(dataList.toString());
        //System.out.println(i);

    }

}
