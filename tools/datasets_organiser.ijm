src = getDirectory("current") + 'datasets-png/';
dst = getDirectory("current") + 'datasets/';
name1 = 'ctc-hela-72im/';
name2 = 'train';
name3 = 'sources';

names3 = newArray("sources", "labels");

File.makeDirectory(dst+name1);
File.makeDirectory(dst+name1+'/'+name2);

for(n=0; n<2; n++) {
	name3 = names3[n];
	File.makeDirectory(dst+name1+'/'+name2+'/'+name3);
	file = File.open(dst + name1 + 'stats-' + name2 + '-' + name3 + '.csv');
	print(file, "Name, Area, Mean, Minimum, Maximum, SDev\n" );

	destin = dst + name1+'/'+name2+'/'+name3+'/';
	source = src + name1+'/'+name2+'/'+name3+'/';
	print(source);
	list = getFileList(source);
	print('Number of files ', list.length);

	for (i = 0; i < list.length; i++) {
    	if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".png")) {
        	open(source + list[i]);
        	title = getTitle();
        	title = substring(title,0,lengthOf(title)-4);
        	getStatistics(area, mean, min, max, std);
	       	print(file, "" + list[i] + ", " + area + ", " + mean + ", " + min + ", " + max + ", " + std);
        	saveAs('tif', destin + title + '.tif');
        	close();
    	}
	}
	File.close(file);
}