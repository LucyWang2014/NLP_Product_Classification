def stitch_files(basename):
    datadir = '../data/'
    featuredir = datadir+basename+'/'
    index_list = []
    for root, dirs, files in os.walk(datadir):
        for fname in files:
            idx_range = get_indexes(fname)
            indexes.append(idx_range[1])
            plog("loading %s..." %fname)
            if fname==basename+'_0_10000.pkl':
                with open(datadir + fname) as f:
                    df=pkl.load(f)
            else:
                with open(datadir + fname) as f:
                    df2=pkl.load(f)
                    df=pd.concat([df,df2])
    max_index = max(indexes)
    outname = datadir + basename + '_0_%s.pkl'%max_index

    plog("writing to file")
    with outname as f:
        pkl.dump(f)
        


if __name__ == '__main__':
    stitch_files('train_image_features')