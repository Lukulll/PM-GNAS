def get_num_features(dataset):
    if dataset == 'Computers':
      return 767
    elif dataset == 'Photo':
      return 745
    elif dataset == 'CS':
      return 6805
    elif dataset == 'Physics':
      return 8415
    elif dataset == 'Cora':
      return 1433
    elif dataset == 'CiteSeer':
      return 3703
    elif dataset == 'PubMed':
      return 500
    else:
       print('update soon')
    return None

def get_num_classes(dataset):
    if dataset == 'Computers':
      return 10
    elif dataset == 'Photo':
      return 8
    elif dataset == 'CS':
      return 15
    elif dataset == 'Physics':
      return 5
    elif dataset == 'Cora':
      return 7
    elif dataset == 'CiteSeer':
      return 6
    elif dataset == 'PubMed':
      return 3
    else:
       print('update soon')
    return None