package ml.shifu.plugin.spark.stats;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.plugin.spark.stats.factory.ColumnStateFactory;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;

import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ColumnStateTest {

	List<Object> oList;
	ColumnState state;
	
	@BeforeTest
	public void createLists() {
		oList= new ArrayList<Object>();
		oList.add(1);
		oList.add("hello");
		oList.add(1.2);
		oList.add("world");
		oList.add(new SerializedCategoricalValueObject());
		state= ColumnStateFactory.getMockColumnState("mock", null, null);
		for(Object obj: oList)
			state.addData(obj);
	}
	
	@Test
	public void testList() {
		//List<Object> otherList= state.getStates().get(0).
	}
}
