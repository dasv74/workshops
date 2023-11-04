src = getDirectory("current") + '../datasets/';
dst = getDirectory("current") + '../datasets/
name1 = 'ctc-glioblastoma-66im/';
name2 = 'train';
names3 = newArray("masks");

File.makeDirectory(dst+name1);
File.makeDirectory(dst+name1+'/'+name2);

for(n=0; n<1; n++) {
	name3 = names3[n];
	destin = dst + name1 + '/'+name2 + '/edges/';
	source = src + name1 + '/'+name2 + '/'+name3+'/';
	
	File.makeDirectory(destin);
	file = File.open(dst + name1 + 'stats-' + name2 + '-edges.csv');
	print(file, "Name, Area, Mean, Minimum, Maximum, SDev\n" );
	print(source);
	list = getFileList(source);
	print('Number of files ', list.length);

	for (i = 0; i < list.length; i++) {
    	if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".png")) {
        	open(source + list[i]);
        	title = getTitle();
        	title = substring(title,0,lengthOf(title)-4);
        	run("Find Edges");
			run("Max...", "value=1");
        	getStatistics(area, mean, min, max, std);
	       	print(file, "" + list[i] + ", " + area + ", " + mean + ", " + min + ", " + max + ", " + std);
        	saveAs('tif', destin + title + '.tif');
        	close();
    	}
	}
	File.close(file);
}