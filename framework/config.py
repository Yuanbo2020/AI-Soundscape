import torch, os

####################################################################################################

cuda = 1

training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

mel_bins = 64

event_labels = ['Aircraft', 'Bells', 'Bird tweet', 'Bus', 'Car',
                'Children', 'Construction', 'Dog bark', 'Footsteps', 'General traffic',
                'Horn', 'Laughter', 'Motorcycle', 'Music',  'Non-identifiable',
                'Other', 'Rail', 'Rustling leaves', 'Screeching brakes', 'Shouting',
                'Siren', 'Speech', 'Ventilation', 'Water']

endswith = '.pth'

cuda_seed = None
