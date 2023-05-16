N = 100;
nepochs = 30;
lrs = nan(N, 1);
vloss = nan(nepochs+1, N);
tloss = nan(nepochs, N);
for idx = 1:N
    try
        load(sprintf('models/lrtune_%03d/summary.mat', idx))
        vloss(:,idx) = cell2mat(val_loss);
        tloss(:,idx) = cell2mat(train_loss);
        lrs(idx) = lr;
    end
end

[~, sidx] = sort(lrs);
vloss = vloss(:,sidx);
tloss = tloss(:,sidx);
lrs = lrs(sidx);
