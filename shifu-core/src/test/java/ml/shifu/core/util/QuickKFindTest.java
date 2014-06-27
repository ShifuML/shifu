package ml.shifu.core.util;

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
