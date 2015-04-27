#### Running Sample

```bash
cd lib
g++ `pkg-config --cflags --libs opencv` surf_flann_matcher.cpp -o surf_flann_matcher
./surf_flann_matcher ../data/query-image.jpg ../data/image1.jpg
```

## Links

* http://answers.opencv.org/question/10459/surf-matching-against-a-database-of-images/
